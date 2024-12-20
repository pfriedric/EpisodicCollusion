import math
from enum import IntEnum
from typing import Any, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from gymnax.environments import environment, spaces

GRID_SIZE = 8
OBS_SIZE = 5
PADDING = OBS_SIZE - 1
NUM_TYPES = 5  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COINS = 6  # per type
NUM_COIN_TYPES = 2
NUM_OBJECTS = 2 + NUM_COIN_TYPES * NUM_COINS + 1  # red, blue, 2 red coin, 2 blue coin

INTERACT_THRESHOLD = 0


@chex.dataclass
class EnvState:
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    red_inventory: jnp.ndarray
    blue_inventory: jnp.ndarray
    red_coins: jnp.ndarray
    blue_coins: jnp.ndarray
    freeze: int


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    interact = 3
    stay = 4


class Items(IntEnum):
    empty = 0
    red_agent = 1
    blue_agent = 2
    red_coin = 3
    blue_coin = 4
    wall = 5
    interact = 6


ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # forward
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap`
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [0, 1, 0],  # up
        [1, 0, 0],  # right
        [0, -1, 0],  # down
        [-1, 0, 0],  # left
    ],
    dtype=jnp.int8,
)

GRID = jnp.zeros(
    (GRID_SIZE + 2 * PADDING, GRID_SIZE + 2 * PADDING),
    dtype=jnp.int8,
)

# First layer of Padding is Wall
GRID = GRID.at[PADDING - 1, :].set(5)
GRID = GRID.at[GRID_SIZE + PADDING, :].set(5)
GRID = GRID.at[:, PADDING - 1].set(5)
GRID = GRID.at[:, GRID_SIZE + PADDING].set(5)

COIN_SPAWNS = [
    [1, 1],
    [1, 2],
    [2, 1],
    [1, GRID_SIZE - 2],
    [2, GRID_SIZE - 2],
    [1, GRID_SIZE - 3],
    # [2, 2],
    # [2, GRID_SIZE - 3],
    [GRID_SIZE - 2, 2],
    [GRID_SIZE - 3, 1],
    [GRID_SIZE - 2, 1],
    [GRID_SIZE - 2, GRID_SIZE - 2],
    [GRID_SIZE - 2, GRID_SIZE - 3],
    [GRID_SIZE - 3, GRID_SIZE - 2],
    # [GRID_SIZE - 3, 2],
    # [GRID_SIZE - 3, GRID_SIZE - 3],
]

COIN_SPAWNS = jnp.array(
    COIN_SPAWNS,
    dtype=jnp.int8,
)

RED_SPAWN = jnp.array(
    COIN_SPAWNS[::2, :],
    dtype=jnp.int8,
)

BLUE_SPAWN = jnp.array(
    COIN_SPAWNS[1::2, :],
    dtype=jnp.int8,
)

AGENT_SPAWNS = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    # [1, 1],
    [1, 2],
    [0, GRID_SIZE - 1],
    [0, GRID_SIZE - 2],
    [0, GRID_SIZE - 3],
    [1, GRID_SIZE - 1],
    # [1, GRID_SIZE - 2],
    # [1, GRID_SIZE - 3],
    [GRID_SIZE - 1, 0],
    [GRID_SIZE - 1, 1],
    [GRID_SIZE - 1, 2],
    [GRID_SIZE - 2, 0],
    # [GRID_SIZE - 2, 1],
    # [GRID_SIZE - 2, 2],
    [GRID_SIZE - 1, GRID_SIZE - 1],
    [GRID_SIZE - 1, GRID_SIZE - 2],
    [GRID_SIZE - 1, GRID_SIZE - 3],
    [GRID_SIZE - 2, GRID_SIZE - 1],
    # [GRID_SIZE - 2, GRID_SIZE - 2],
    [GRID_SIZE - 2, GRID_SIZE - 3],
]

AGENT_SPAWNS = jnp.array(
    [[(j, i), (GRID_SIZE - 1 - j, GRID_SIZE - 1 - i)] for (i, j) in AGENT_SPAWNS],
    dtype=jnp.int8,
).reshape(-1, 2, 2)


PLAYER1_COLOUR = (255.0, 127.0, 14.0)
PLAYER2_COLOUR = (31.0, 119.0, 180.0)
GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)


class InTheMatrix(environment.Environment):
    """
    JAX Compatible version of *inTheMatix environment.
    """

    # used for caching
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps: int,
        num_outer_steps: int,
        fixed_coin_location: bool,
    ):

        super().__init__()

        def _get_obs_point(x: int, y: int, dir: int) -> jnp.ndarray:
            x, y = x + PADDING, y + PADDING
            x = jnp.where(dir == 0, x - (OBS_SIZE // 2), x)
            x = jnp.where(dir == 2, x - (OBS_SIZE // 2), x)
            x = jnp.where(dir == 3, x - (OBS_SIZE - 1), x)

            y = jnp.where(dir == 1, y - (OBS_SIZE // 2), y)
            y = jnp.where(dir == 2, y - (OBS_SIZE - 1), y)
            y = jnp.where(dir == 3, y - (OBS_SIZE // 2), y)
            return x, y

        def _get_obs(state: EnvState) -> jnp.ndarray:
            # create state
            grid = jnp.pad(
                state.grid,
                ((PADDING, PADDING), (PADDING, PADDING)),
                constant_values=Items.wall,
            )
            x, y = _get_obs_point(state.red_pos[0], state.red_pos[1], state.red_pos[2])
            grid1 = jax.lax.dynamic_slice(
                grid,
                start_indices=(x, y),
                slice_sizes=(OBS_SIZE, OBS_SIZE),
            )
            # rotate
            grid1 = jnp.where(
                state.red_pos[2] == 1,
                jnp.rot90(grid1, k=1, axes=(0, 1)),
                grid1,
            )
            grid1 = jnp.where(
                state.red_pos[2] == 2,
                jnp.rot90(grid1, k=2, axes=(0, 1)),
                grid1,
            )
            grid1 = jnp.where(
                state.red_pos[2] == 3,
                jnp.rot90(grid1, k=3, axes=(0, 1)),
                grid1,
            )

            angle1 = -1 * jnp.ones_like(grid1, dtype=jnp.int8)
            angle1 = jnp.where(
                grid1 == Items.blue_agent,
                (state.blue_pos[2] - state.red_pos[2]) % 4,
                -1,
            )
            angle1 = jax.nn.one_hot(angle1, 4)

            # one-hot (drop first channel as its empty blocks)
            grid1 = jax.nn.one_hot(grid1 - 1, len(Items) - 1, dtype=jnp.int8)
            obs1 = jnp.concatenate([grid1, angle1], axis=-1)

            x, y = _get_obs_point(
                state.blue_pos[0], state.blue_pos[1], state.blue_pos[2]
            )

            grid2 = jax.lax.dynamic_slice(
                grid,
                start_indices=(x, y),
                slice_sizes=(OBS_SIZE, OBS_SIZE),
            )

            grid2 = jnp.where(
                state.blue_pos[2] == 1,
                jnp.rot90(grid2, k=1, axes=(0, 1)),
                grid2,
            )
            grid2 = jnp.where(
                state.blue_pos[2] == 2,
                jnp.rot90(grid2, k=2, axes=(0, 1)),
                grid2,
            )
            grid2 = jnp.where(
                state.blue_pos[2] == 3,
                jnp.rot90(grid2, k=3, axes=(0, 1)),
                grid2,
            )

            angle2 = -1 * jnp.ones_like(grid2, dtype=jnp.int8)
            angle2 = jnp.where(
                grid2 == Items.red_agent,
                (state.red_pos[2] - state.blue_pos[2]) % 4,
                -1,
            )
            angle2 = jax.nn.one_hot(angle2, 4)

            # sends 0 -> -1 and droped by one_hot
            grid2 = jax.nn.one_hot(grid2 - 1, len(Items) - 1, dtype=jnp.int8)
            # make agent 2 think it is agent 1
            _grid2 = grid2.at[:, :, 0].set(grid2[:, :, 1])
            _grid2 = _grid2.at[:, :, 1].set(grid2[:, :, 0])
            _obs2 = jnp.concatenate([_grid2, angle2], axis=-1)

            red_pickup = jnp.sum(state.red_inventory) > INTERACT_THRESHOLD
            blue_pickup = jnp.sum(state.blue_inventory) > INTERACT_THRESHOLD

            blue_to_show = jnp.where(state.freeze >= 0, state.blue_inventory, 0)
            red_to_show = jnp.where(state.freeze >= 0, state.red_inventory, 0)

            return {
                "observation": obs1,
                "inventory": jnp.array(
                    [
                        state.red_inventory[0],
                        state.red_inventory[1],
                        red_pickup,
                        blue_pickup,
                        blue_to_show[0],
                        blue_to_show[1],
                    ],
                    dtype=jnp.int8,
                ),
            }, {
                "observation": _obs2,
                "inventory": jnp.array(
                    [
                        state.blue_inventory[0],
                        state.blue_inventory[1],
                        blue_pickup,
                        red_pickup,
                        red_to_show[0],
                        red_to_show[1],
                    ],
                    dtype=jnp.int8,
                ),
            }

        def _get_reward(state: EnvState, params: EnvParams) -> jnp.ndarray:
            inv1 = state.red_inventory / state.red_inventory.sum()
            inv2 = state.blue_inventory / state.blue_inventory.sum()
            r1 = inv1 @ params.payoff_matrix[0] @ inv2.T
            r2 = inv1 @ params.payoff_matrix[1] @ inv2.T
            return r1, r2

        def _interact(
            state: EnvState, actions: Tuple[int, int], params: EnvParams
        ) -> Tuple[bool, float, float, EnvState]:
            # if interact
            a0, a1 = actions

            red_zap = a0 == Actions.interact
            blue_zap = a1 == Actions.interact
            interact_idx = jnp.int8(Items.interact)

            # remove old interacts
            state.grid = jnp.where(
                state.grid == interact_idx, jnp.int8(Items.empty), state.grid
            )

            # check 1 ahead
            red_target = jnp.clip(
                state.red_pos + STEP[state.red_pos[2]], 0, GRID_SIZE - 1
            )
            blue_target = jnp.clip(
                state.blue_pos + STEP[state.blue_pos[2]], 0, GRID_SIZE - 1
            )

            red_interact = state.grid[red_target[0], red_target[1]] == Items.blue_agent
            blue_interact = (
                state.grid[blue_target[0], blue_target[1]] == Items.red_agent
            )

            # check 2 ahead
            red_target_ahead = jnp.clip(
                state.red_pos + 2 * STEP[state.red_pos[2]], 0, GRID_SIZE - 1
            )
            blue_target_ahead = jnp.clip(
                state.blue_pos + 2 * STEP[state.blue_pos[2]], 0, GRID_SIZE - 1
            )

            red_interact_ahead = (
                state.grid[red_target_ahead[0], red_target_ahead[1]] == Items.blue_agent
            )
            blue_interact_ahead = (
                state.grid[blue_target_ahead[0], blue_target_ahead[1]]
                == Items.red_agent
            )

            # check to your right  - clip can't be used here as it will wrap down
            red_target_right = (
                state.red_pos
                + STEP[state.red_pos[2]]
                + STEP[(state.red_pos[2] + 1) % 4]
            )
            oob_red = jnp.logical_or(
                (red_target_right > GRID_SIZE - 1).any(),
                (red_target_right < 0).any(),
            )
            red_target_right = jnp.where(oob_red, red_target, red_target_right)

            blue_target_right = (
                state.blue_pos
                + STEP[state.blue_pos[2]]
                + STEP[(state.blue_pos[2] + 1) % 4]
            )
            oob_blue = jnp.logical_or(
                (blue_target_right > GRID_SIZE - 1).any(),
                (blue_target_right < 0).any(),
            )
            blue_target_right = jnp.where(oob_blue, blue_target, blue_target_right)

            red_interact_right = (
                state.grid[red_target_right[0], red_target_right[1]] == Items.blue_agent
            )
            blue_interact_right = (
                state.grid[blue_target_right[0], blue_target_right[1]]
                == Items.red_agent
            )

            # check to your left
            red_target_left = (
                state.red_pos
                + STEP[state.red_pos[2]]
                + STEP[(state.red_pos[2] - 1) % 4]
            )
            oob_red = jnp.logical_or(
                (red_target_left > GRID_SIZE - 1).any(),
                (red_target_left < 0).any(),
            )
            red_target_left = jnp.where(oob_red, red_target, red_target_left)

            blue_target_left = (
                state.blue_pos
                + STEP[state.blue_pos[2]]
                + STEP[(state.blue_pos[2] - 1) % 4]
            )
            oob_blue = jnp.logical_or(
                (blue_target_left > GRID_SIZE - 1).any(),
                (blue_target_left < 0).any(),
            )
            blue_target_left = jnp.where(oob_blue, blue_target, blue_target_left)

            red_interact_left = (
                state.grid[red_target_left[0], red_target_left[1]] == Items.blue_agent
            )
            blue_interact_left = (
                state.grid[blue_target_left[0], blue_target_left[1]] == Items.red_agent
            )

            red_interact = jnp.logical_or(
                red_interact,
                jnp.logical_or(
                    red_interact_ahead,
                    jnp.logical_or(red_interact_right, red_interact_left),
                ),
            )

            # update grid with red zaps
            aux_grid = jnp.copy(state.grid)

            item = jnp.where(
                state.grid[red_target[0], red_target[1]],
                state.grid[red_target[0], red_target[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target[0], red_target[1]].set(item)

            item = jnp.where(
                state.grid[red_target_ahead[0], red_target_ahead[1]],
                state.grid[red_target_ahead[0], red_target_ahead[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target_ahead[0], red_target_ahead[1]].set(item)

            item = jnp.where(
                state.grid[red_target_right[0], red_target_right[1]],
                state.grid[red_target_right[0], red_target_right[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target_right[0], red_target_right[1]].set(item)

            item = jnp.where(
                state.grid[red_target_left[0], red_target_left[1]],
                state.grid[red_target_left[0], red_target_left[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target_left[0], red_target_left[1]].set(item)

            state.grid = jnp.where(red_zap, aux_grid, state.grid)

            # update grid with blue zaps
            aux_grid = jnp.copy(state.grid)
            blue_interact = jnp.logical_or(
                blue_interact,
                jnp.logical_or(
                    blue_interact_ahead,
                    jnp.logical_or(blue_interact_right, blue_interact_left),
                ),
            )
            item = jnp.where(
                state.grid[blue_target[0], blue_target[1]],
                state.grid[blue_target[0], blue_target[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[blue_target[0], blue_target[1]].set(item)

            item = jnp.where(
                state.grid[blue_target_ahead[0], blue_target_ahead[1]],
                state.grid[blue_target_ahead[0], blue_target_ahead[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[blue_target_ahead[0], blue_target_ahead[1]].set(item)

            item = jnp.where(
                state.grid[blue_target_right[0], blue_target_right[1]],
                state.grid[blue_target_right[0], blue_target_right[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[blue_target_right[0], blue_target_right[1]].set(item)

            item = jnp.where(
                state.grid[blue_target_left[0], blue_target_left[1]],
                state.grid[blue_target_left[0], blue_target_left[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[blue_target_left[0], blue_target_left[1]].set(item)
            state.grid = jnp.where(blue_zap, aux_grid, state.grid)

            # rewards
            red_reward, blue_reward = 0.0, 0.0
            _r_reward, _b_reward = _get_reward(state, params)

            interact = jnp.logical_or(red_zap * red_interact, blue_zap * blue_interact)

            red_pickup = state.red_inventory.sum() > INTERACT_THRESHOLD
            blue_pickup = state.blue_inventory.sum() > INTERACT_THRESHOLD
            interact = jnp.logical_and(
                interact, jnp.logical_and(red_pickup, blue_pickup)
            )

            red_reward = jnp.where(interact, red_reward + _r_reward, red_reward)
            blue_reward = jnp.where(interact, blue_reward + _b_reward, blue_reward)
            return interact, red_reward, blue_reward, state

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            """Step the environment."""

            # freeze check
            action_0, action_1 = actions
            action_0 = jnp.where(state.freeze > 0, Actions.stay, action_0)
            action_1 = jnp.where(state.freeze > 0, Actions.stay, action_1)

            # turning red
            new_red_pos = jnp.int8(
                (state.red_pos + ROTATIONS[action_0])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4])
            )

            # moving red
            red_move = action_0 == Actions.forward
            new_red_pos = jnp.where(
                red_move, new_red_pos + STEP[state.red_pos[2]], new_red_pos
            )
            new_red_pos = jnp.clip(
                new_red_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array([GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8),
            )

            # if you bounced back to ur original space, we change your move to stay (for collision logic)
            red_move = (new_red_pos[:2] != state.red_pos[:2]).any()

            # turning blue
            new_blue_pos = jnp.int8(
                (state.blue_pos + ROTATIONS[action_1])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4], dtype=jnp.int8)
            )

            # moving blue
            blue_move = action_1 == Actions.forward
            new_blue_pos = jnp.where(
                blue_move, new_blue_pos + STEP[state.blue_pos[2]], new_blue_pos
            )
            new_blue_pos = jnp.clip(
                new_blue_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array([GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8),
            )
            blue_move = (new_blue_pos[:2] != state.blue_pos[:2]).any()

            # if collision, priority to whoever didn't move
            collision = jnp.all(new_red_pos[:2] == new_blue_pos[:2])

            new_red_pos = jnp.where(
                collision * red_move * (1 - blue_move),  # red moved, blue didn't
                state.red_pos,
                new_red_pos,
            )
            new_blue_pos = jnp.where(
                collision * (1 - red_move) * blue_move,  # blue moved, red didn't
                state.blue_pos,
                new_blue_pos,
            )

            # if both moved, then randomise
            red_takes_square = jax.random.choice(key, jnp.array([0, 1]))
            new_red_pos = jnp.where(
                collision
                * blue_move
                * red_move
                * (1 - red_takes_square),  # if both collide and red doesn't take square
                state.red_pos,
                new_red_pos,
            )
            new_blue_pos = jnp.where(
                collision
                * blue_move
                * red_move
                * (red_takes_square),  # if both collide and blue doesn't take square
                state.blue_pos,
                new_blue_pos,
            )

            # update inventories
            red_red_matches = (
                state.grid[new_red_pos[0], new_red_pos[1]] == Items.red_coin
            )
            red_blue_matches = (
                state.grid[new_red_pos[0], new_red_pos[1]] == Items.blue_coin
            )
            blue_red_matches = (
                state.grid[new_blue_pos[0], new_blue_pos[1]] == Items.red_coin
            )
            blue_blue_matches = (
                state.grid[new_blue_pos[0], new_blue_pos[1]] == Items.blue_coin
            )

            state.red_inventory = state.red_inventory + jnp.array(
                [red_red_matches, red_blue_matches]
            )
            state.blue_inventory = state.blue_inventory + jnp.array(
                [blue_red_matches, blue_blue_matches]
            )

            # update grid
            state.grid = state.grid.at[(state.red_pos[0], state.red_pos[1])].set(
                jnp.int8(Items.empty)
            )
            state.grid = state.grid.at[(state.blue_pos[0], state.blue_pos[1])].set(
                jnp.int8(Items.empty)
            )
            state.grid = state.grid.at[(new_red_pos[0], new_red_pos[1])].set(
                jnp.int8(Items.red_agent)
            )
            state.grid = state.grid.at[(new_blue_pos[0], new_blue_pos[1])].set(
                jnp.int8(Items.blue_agent)
            )
            state.red_pos = new_red_pos
            state.blue_pos = new_blue_pos

            red_reward, blue_reward = 0, 0
            (
                interact,
                red_interact_reward,
                blue_interact_reward,
                state,
            ) = _interact(state, (action_0, action_1), params)
            red_reward += red_interact_reward
            blue_reward += blue_interact_reward

            # if we interacted, then we set freeze
            state.freeze = jnp.where(interact, params.freeze_penalty, state.freeze)

            # if we didn't interact, then we decrement freeze
            state.freeze = jnp.where(state.freeze > 0, state.freeze - 1, state.freeze)
            state_sft_re = _soft_reset_state(key, state)
            state = jax.tree_map(
                lambda x, y: jnp.where(state.freeze == 0, x, y),
                state_sft_re,
                state,
            )
            state_nxt = EnvState(
                red_pos=state.red_pos,
                blue_pos=state.blue_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                red_inventory=state.red_inventory,
                blue_inventory=state.blue_inventory,
                red_coins=state.red_coins,
                blue_coins=state.blue_coins,
                freeze=jnp.where(interact, params.freeze_penalty, state.freeze),
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key, params)
            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree_map(
                lambda x, y: jax.lax.select(reset_inner, x, y),
                state_re,
                state_nxt,
            )

            obs = _get_obs(state)
            blue_reward = jnp.where(reset_inner, 0, blue_reward)
            red_reward = jnp.where(reset_inner, 0, red_reward)
            return (
                obs,
                state,
                (red_reward, blue_reward),
                reset_inner,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _soft_reset_state(key: jnp.ndarray, state: EnvState) -> EnvState:
            """Reset the grid to original state and"""
            # Find the free spaces in the grid
            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)

            # if coin location can change, then we need to reset the coins
            for i in range(NUM_COINS):
                grid = grid.at[state.red_coins[i, 0], state.red_coins[i, 1]].set(
                    jnp.int8(Items.red_coin)
                )

            for i in range(NUM_COINS):
                grid = grid.at[state.blue_coins[i, 0], state.blue_coins[i, 1]].set(
                    jnp.int8(Items.blue_coin)
                )

            agent_pos = jax.random.choice(key, AGENT_SPAWNS, shape=(), replace=False)

            player_dir = jax.random.randint(
                key, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array([agent_pos[:2, 0], agent_pos[:2, 1], player_dir]).T

            red_pos = player_pos[0, :]
            blue_pos = player_pos[1, :]

            grid = grid.at[red_pos[0], red_pos[1]].set(jnp.int8(Items.red_agent))
            grid = grid.at[blue_pos[0], blue_pos[1]].set(jnp.int8(Items.blue_agent))

            return EnvState(
                red_pos=red_pos,
                blue_pos=blue_pos,
                inner_t=state.inner_t,
                outer_t=state.outer_t,
                grid=grid,
                red_inventory=jnp.zeros(2),
                blue_inventory=jnp.zeros(2),
                red_coins=state.red_coins,
                blue_coins=state.blue_coins,
                freeze=jnp.int16(-1),
            )

        def _reset_state(
            key: jnp.ndarray, params: EnvParams
        ) -> Tuple[jnp.ndarray, EnvState]:
            key, subkey = jax.random.split(key)

            # coin_pos = jax.random.choice(
            #     subkey, COIN_SPAWNS, shape=(NUM_COIN_TYPES*NUM_COINS,), replace=False
            # )

            agent_pos = jax.random.choice(subkey, AGENT_SPAWNS, shape=(), replace=False)
            player_dir = jax.random.randint(
                subkey, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array([agent_pos[:2, 0], agent_pos[:2, 1], player_dir]).T
            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)
            grid = grid.at[player_pos[0, 0], player_pos[0, 1]].set(
                jnp.int8(Items.red_agent)
            )
            grid = grid.at[player_pos[1, 0], player_pos[1, 1]].set(
                jnp.int8(Items.blue_agent)
            )
            if fixed_coin_location:
                rand_idx = jax.random.randint(subkey, shape=(), minval=0, maxval=1)
                red_coins = jnp.where(rand_idx, RED_SPAWN, BLUE_SPAWN)
                blue_coins = jnp.where(rand_idx, BLUE_SPAWN, RED_SPAWN)
            else:
                coin_spawn = jax.random.permutation(subkey, COIN_SPAWNS, axis=0)
                red_coins = coin_spawn[:NUM_COINS, :]
                blue_coins = coin_spawn[NUM_COINS:, :]

            for i in range(NUM_COINS):
                grid = grid.at[red_coins[i, 0], red_coins[i, 1]].set(
                    jnp.int8(Items.red_coin)
                )

            for i in range(NUM_COINS):
                grid = grid.at[blue_coins[i, 0], blue_coins[i, 1]].set(
                    jnp.int8(Items.blue_coin)
                )

            return EnvState(
                red_pos=player_pos[0, :],
                blue_pos=player_pos[1, :],
                inner_t=0,
                outer_t=0,
                grid=grid,
                red_inventory=jnp.zeros(2),
                blue_inventory=jnp.zeros(2),
                red_coins=red_coins,
                blue_coins=blue_coins,
                freeze=jnp.int16(-1),
            )

        def reset(key: jnp.ndarray, params: EnvParams) -> Tuple[jnp.ndarray, EnvState]:
            state = _reset_state(key, params)
            obs = _get_obs(state)
            return obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        # self.step = jax.jit(_step)
        self.step = _step
        self.reset = jax.jit(reset)
        self.get_obs_point = _get_obs_point
        self.get_reward = _get_reward

        # for debugging
        self.get_obs = _get_obs
        self.cnn = True

        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

    @property
    def name(self) -> str:
        """Environment name."""
        return "CoinGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self, params: EnvParams) -> spaces.Dict:
        """Observation space of the environment."""
        _shape = (
            (OBS_SIZE, OBS_SIZE, len(Items) - 1 + 4)
            if self.cnn
            else (OBS_SIZE**2 * (len(Items) - 1 + 4),)
        )

        return {
            "observation": spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8),
            "inventory": spaces.Box(
                low=0,
                high=NUM_COINS,
                shape=NUM_COIN_TYPES + 4,
                dtype=jnp.uint8,
            ),
        }

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (GRID_SIZE, GRID_SIZE, NUM_TYPES + 4)
            if self.cnn
            else (GRID_SIZE**2 * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)


if __name__ == "__main__":
    from PIL import Image

    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    print(f"Jax backend: {jax.lib.xla_bridge.get_backend().platform}")

    action = 1
    num_outer_steps = 1
    num_inner_steps = 150
    render_agent_view = False

    rng = jax.random.PRNGKey(0)
    env = InTheMatrix(num_inner_steps, num_outer_steps, True)
    num_actions = env.action_space().n
    params = EnvParams(
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,
    )
    obs, old_state = env.reset(rng, params)

    print(env.observation_space(params))
    obs_shape = jax.tree_map(lambda x: x.shape, env.observation_space(params))
    print(obs_shape)

    # print(obs)
    # pics = []
    # pics1 = []
    # pics2 = []

    # int_action = {
    #     0: "left",
    #     1: "right",
    #     2: "forward",
    #     3: "interact",
    #     4: "stay",
    # }

    # key_int = {"w": 2, "a": 0, "s": 4, "d": 1, " ": 4}
    # env.step = jax.jit(env.step)

    # for t in range(num_outer_steps * num_inner_steps):
    #     rng, rng1, rng2 = jax.random.split(rng, 3)
    #     # a1 = jnp.array(2)
    #     # a2 = jnp.array(4)
    #     a1 = jax.random.choice(
    #         rng1, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
    #     )
    #     a2 = jax.random.choice(
    #         rng2, a=num_actions, p=jnp.array([0.1, 0.1, 0.5, 0.1, 0.4])
    #     )
    #     obs, state, reward, done, info = env.step(
    #         rng, old_state, (a1 * action, a2 * action), params
    #     )

    #     if (state.red_pos[:2] == state.blue_pos[:2]).all():
    #         import pdb

    #         # pdb.set_trace()
    #         print("collision")
    #         print(
    #             f"timestep: {t}, A1: {int_action[a1.item()]} A2:{int_action[a2.item()]}"
    #         )
    #         print(state.red_pos, state.blue_pos)
