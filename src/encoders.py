"""
State and Action encoders for AlphaZero Catan.

Proper encoding is CRITICAL for the neural network to learn.
Each unique game state and action must map to a unique index.
"""
import numpy as np
from catanatron import Color, Game

# =============================================================================
# CONSTANTS - Catan board dimensions
# =============================================================================
NUM_NODES = 54          # Settlement/city locations
NUM_TILES = 19          # Hex tiles
NUM_PLAYERS = 4         # Max players
NUM_RESOURCES = 5       # WOOD, BRICK, SHEEP, WHEAT, ORE

# Resource type to index mapping
RESOURCE_MAP = {
    'WOOD': 0,
    'BRICK': 1,
    'SHEEP': 2,
    'WHEAT': 3,
    'ORE': 4,
    None: 5,  # Desert
}

RESOURCE_LIST = ['WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']

# Dice number to probability (dots on the number token)
NUMBER_PROBS = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
    7: 0,  # Robber
    8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
    None: 0,  # Desert
}


# =============================================================================
# EDGE MAPPING - Actual 72 edges discovered from catanatron gameplay
# =============================================================================
# These are the actual edges (node pairs) used in catanatron's board representation
EDGE_LIST = [
    (0, 1), (0, 5), (0, 20), (1, 2), (1, 6), (2, 3), (2, 9), (3, 4), (3, 12),
    (4, 5), (4, 15), (5, 16), (6, 7), (6, 23), (7, 8), (7, 24), (8, 9), (8, 27),
    (9, 10), (10, 11), (10, 29), (11, 12), (11, 32), (12, 13), (13, 14), (13, 34),
    (14, 15), (14, 37), (15, 17), (16, 18), (16, 21), (17, 18), (17, 39), (18, 40),
    (19, 20), (19, 21), (19, 46), (20, 22), (21, 43), (22, 23), (22, 49), (23, 52),
    (24, 25), (24, 53), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
    (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39),
    (39, 41), (40, 42), (40, 44), (41, 42), (43, 44), (43, 47), (45, 46), (45, 47),
    (46, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53),
]

EDGE_TO_IDX = {edge: i for i, edge in enumerate(EDGE_LIST)}
NUM_EDGES = len(EDGE_LIST)  # Exactly 72


# =============================================================================
# STATE ENCODER
# =============================================================================
class StateEncoder:
    """
    Encodes Catan game state into a fixed-size feature vector.
    """

    def __init__(self, num_players: int = 4):
        self.num_players = num_players

        # Calculate feature dimensions
        self.tile_features = NUM_TILES * 7  # 6 resource types + 1 prob
        self.node_features = NUM_NODES * (self.num_players * 2 + 1)  # owner×type + empty
        self.edge_features = NUM_EDGES * (self.num_players + 1)  # owner + empty
        self.player_features = self.num_players * (5 + 5 + 1 + 1 + 1 + 6)  # res, dev, vp, army, road, ports
        self.robber_features = NUM_TILES
        self.current_player_features = self.num_players
        self.phase_features = 3

        self.state_size = (
            self.tile_features +
            self.node_features +
            self.edge_features +
            self.player_features +
            self.robber_features +
            self.current_player_features +
            self.phase_features
        )

    def encode(self, game: Game) -> np.ndarray:
        """Encode full game state to feature vector."""
        state = game.state
        board = state.board
        ps = state.player_state

        features = []

        # 1. TILE FEATURES
        tile_feats = self._encode_tiles(board.map)
        features.extend(tile_feats)

        # 2. NODE FEATURES (buildings)
        node_feats = self._encode_nodes(board)
        features.extend(node_feats)

        # 3. EDGE FEATURES (roads)
        edge_feats = self._encode_edges(board)
        features.extend(edge_feats)

        # 4. PLAYER FEATURES
        for p_idx in range(self.num_players):
            player_feats = self._encode_player(ps, p_idx, board, game.state.colors[p_idx])
            features.extend(player_feats)

        # 5. ROBBER POSITION
        robber_feats = self._encode_robber(board)
        features.extend(robber_feats)

        # 6. CURRENT PLAYER
        current_feats = np.zeros(self.num_players)
        current_feats[state.current_player_index] = 1
        features.extend(current_feats)

        # 7. GAME PHASE
        phase_feats = [
            1.0 if state.is_initial_build_phase else 0.0,
            1.0 if state.is_discarding else 0.0,
            1.0 if state.is_moving_knight else 0.0,
        ]
        features.extend(phase_feats)

        return np.array(features, dtype=np.float32)

    def _encode_tiles(self, catan_map) -> list:
        """Encode tile resources and numbers."""
        features = []

        # Sort tiles by coordinate for consistent ordering
        sorted_tiles = sorted(catan_map.land_tiles.items())

        for coord, tile in sorted_tiles:
            # Resource one-hot (6 types including desert/None)
            resource_onehot = [0.0] * 6
            res_idx = RESOURCE_MAP.get(str(tile.resource) if tile.resource else None, 5)
            resource_onehot[res_idx] = 1.0
            features.extend(resource_onehot)

            # Number probability
            prob = NUMBER_PROBS.get(tile.number, 0)
            features.append(prob * 6)  # Scale to ~0-1 range

        return features

    def _encode_nodes(self, board) -> list:
        """Encode buildings at each node."""
        features = []

        for node_id in range(NUM_NODES):
            # One-hot for: empty, P0_settlement, P0_city, P1_settlement, P1_city, ...
            node_onehot = [0.0] * (self.num_players * 2 + 1)

            building = board.buildings.get(node_id)
            if building is None:
                node_onehot[0] = 1.0  # Empty
            else:
                color, building_type = building
                player_idx = self._color_to_idx(color)
                if player_idx is not None:
                    if building_type == 'SETTLEMENT':
                        node_onehot[1 + player_idx * 2] = 1.0
                    else:  # CITY
                        node_onehot[2 + player_idx * 2] = 1.0

            features.extend(node_onehot)

        return features

    def _encode_edges(self, board) -> list:
        """Encode roads at each edge."""
        # Build owner map for edges we know about
        edge_owners = {}  # canonical edge_idx -> player_idx

        for edge, color in board.roads.items():
            if isinstance(edge, tuple) and len(edge) == 2:
                norm_edge = tuple(sorted(edge))
                if norm_edge in EDGE_TO_IDX:
                    edge_idx = EDGE_TO_IDX[norm_edge]
                    edge_owners[edge_idx] = self._color_to_idx(color)

        features = []
        for i in range(NUM_EDGES):
            # One-hot for: empty, P0, P1, P2, P3
            edge_onehot = [0.0] * (self.num_players + 1)

            owner = edge_owners.get(i)
            if owner is None:
                edge_onehot[0] = 1.0  # Empty
            else:
                edge_onehot[1 + owner] = 1.0

            features.extend(edge_onehot)

        return features

    def _encode_player(self, ps: dict, p_idx: int, board, color: Color) -> list:
        """Encode a single player's state."""
        features = []

        # Resources (normalized)
        for res in RESOURCE_LIST:
            count = ps.get(f"P{p_idx}_{res}_IN_HAND", 0)
            features.append(min(count, 10) / 10.0)

        # Development cards (normalized)
        for card in ['KNIGHT', 'YEAR_OF_PLENTY', 'MONOPOLY', 'ROAD_BUILDING', 'VICTORY_POINT']:
            count = ps.get(f"P{p_idx}_{card}_IN_HAND", 0)
            features.append(min(count, 5) / 5.0)

        # Victory points
        vp = ps.get(f"P{p_idx}_VICTORY_POINTS", 0)
        features.append(vp / 10.0)

        # Army size
        army = ps.get(f"P{p_idx}_PLAYED_KNIGHT", 0)
        features.append(min(army, 10) / 10.0)

        # Longest road length
        road = ps.get(f"P{p_idx}_LONGEST_ROAD_LENGTH", 0)
        features.append(min(road, 15) / 15.0)

        # Port access (6 types: 3:1, wood, brick, sheep, wheat, ore)
        port_access = board.get_player_port_resources(color) if color else set()
        port_types = [None, 'WOOD', 'BRICK', 'SHEEP', 'WHEAT', 'ORE']
        for port_type in port_types:
            has_port = port_type in port_access if port_access else False
            features.append(1.0 if has_port else 0.0)

        return features

    def _encode_robber(self, board) -> list:
        """Encode robber position as one-hot over tiles."""
        features = [0.0] * NUM_TILES

        robber_coord = board.robber_coordinate
        # Find index of robber tile
        sorted_tiles = sorted(board.map.land_tiles.keys())
        for i, coord in enumerate(sorted_tiles):
            if coord == robber_coord:
                features[i] = 1.0
                break

        return features

    def _color_to_idx(self, color: Color) -> int:
        """Convert color to player index."""
        color_order = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
        try:
            return color_order.index(color)
        except ValueError:
            return None


# =============================================================================
# ACTION ENCODER
# =============================================================================
class ActionEncoder:
    """
    Encodes Catan actions to unique fixed-size action indices.

    CRITICAL: Every unique action must have a unique index. No collisions!

    Action space layout:
    - ROLL: 1 action (index 0)
    - END_TURN: 1 action (index 1)
    - BUILD_SETTLEMENT: 54 actions (indices 2-55, one per node)
    - BUILD_CITY: 54 actions (indices 56-109, one per node)
    - BUILD_ROAD: NUM_EDGES actions (indices 110-181, one per edge in canonical list)
    - BUY_DEVELOPMENT_CARD: 1 action
    - PLAY_KNIGHT_CARD: 19 × 5 = 95 actions (tile × victim: 4 players + none)
    - MOVE_ROBBER: 19 × 5 = 95 actions (tile × victim: 4 players + none)
    - PLAY_YEAR_OF_PLENTY: 15 actions (resource pairs)
    - PLAY_MONOPOLY: 5 actions (one per resource)
    - PLAY_ROAD_BUILDING: NUM_EDGES × NUM_EDGES = could be huge, simplify to 1
    - MARITIME_TRADE: 5 × 4 = 20 actions (give × receive, must be different)
    - DISCARD: 50 actions (common patterns)
    """

    def __init__(self):
        # Build offsets dynamically
        self._build_offsets()

        # Tile coordinate to index mapping (built at runtime)
        self.tile_to_idx = {}
        self.tile_list = []  # Sorted list of tile coordinates

    def _build_offsets(self):
        """Build action type offsets."""
        offset = 0

        # ROLL: 1
        self.ROLL_START = offset
        offset += 1

        # END_TURN: 1
        self.END_TURN_START = offset
        offset += 1

        # BUILD_SETTLEMENT: 54
        self.BUILD_SETTLEMENT_START = offset
        offset += NUM_NODES

        # BUILD_CITY: 54
        self.BUILD_CITY_START = offset
        offset += NUM_NODES

        # BUILD_ROAD: NUM_EDGES (72)
        self.BUILD_ROAD_START = offset
        offset += NUM_EDGES

        # BUY_DEVELOPMENT_CARD: 1
        self.BUY_DEV_CARD_START = offset
        offset += 1

        # PLAY_KNIGHT_CARD: 19 tiles × 5 victims (4 players + none)
        self.PLAY_KNIGHT_START = offset
        offset += NUM_TILES * 5

        # MOVE_ROBBER: 19 tiles × 5 victims
        self.MOVE_ROBBER_START = offset
        offset += NUM_TILES * 5

        # PLAY_YEAR_OF_PLENTY: 15 pairs (C(5,2) + 5 same-same)
        self.YEAR_OF_PLENTY_START = offset
        offset += 15

        # PLAY_MONOPOLY: 5
        self.MONOPOLY_START = offset
        offset += 5

        # PLAY_ROAD_BUILDING: simplified to 1 (game handles the edges)
        self.ROAD_BUILDING_START = offset
        offset += 1

        # MARITIME_TRADE: 5 give × 4 receive × 3 ratios = 60
        self.MARITIME_TRADE_START = offset
        offset += 60

        # DISCARD: 50 common patterns
        self.DISCARD_START = offset
        offset += 50

        self.action_space_size = offset

    def encode(self, action, game: Game = None) -> int:
        """Convert a Catan action to a unique integer index."""
        action_type = action.action_type.name
        value = action.value

        # Build tile mapping if needed
        if game and not self.tile_to_idx:
            self._build_tile_mapping(game.state.board.map)

        if action_type == 'ROLL':
            return self.ROLL_START

        elif action_type == 'END_TURN':
            return self.END_TURN_START

        elif action_type == 'BUILD_SETTLEMENT':
            node_id = value
            if 0 <= node_id < NUM_NODES:
                return self.BUILD_SETTLEMENT_START + node_id
            return self.BUILD_SETTLEMENT_START  # Fallback

        elif action_type == 'BUILD_CITY':
            node_id = value
            if 0 <= node_id < NUM_NODES:
                return self.BUILD_CITY_START + node_id
            return self.BUILD_CITY_START  # Fallback

        elif action_type == 'BUILD_ROAD':
            # value is (node1, node2) tuple
            edge = tuple(sorted(value))
            if edge in EDGE_TO_IDX:
                return self.BUILD_ROAD_START + EDGE_TO_IDX[edge]
            # Unknown edge - should not happen
            return self.BUILD_ROAD_START

        elif action_type == 'BUY_DEVELOPMENT_CARD':
            return self.BUY_DEV_CARD_START

        elif action_type == 'PLAY_KNIGHT_CARD':
            return self._encode_robber_action(value, self.PLAY_KNIGHT_START)

        elif action_type == 'MOVE_ROBBER':
            return self._encode_robber_action(value, self.MOVE_ROBBER_START)

        elif action_type == 'PLAY_YEAR_OF_PLENTY':
            return self.YEAR_OF_PLENTY_START + self._resource_pair_idx(value)

        elif action_type == 'PLAY_MONOPOLY':
            return self.MONOPOLY_START + self._resource_to_idx(value)

        elif action_type == 'PLAY_ROAD_BUILDING':
            return self.ROAD_BUILDING_START

        elif action_type == 'MARITIME_TRADE':
            return self._encode_maritime_trade(value)

        elif action_type == 'DISCARD':
            return self._encode_discard(value)

        # Unknown action type
        return 0

    def _encode_robber_action(self, value, base_offset: int) -> int:
        """Encode MOVE_ROBBER or PLAY_KNIGHT_CARD action."""
        # value is typically (tile_coord, victim_color, None) or (tile_coord, None, None)
        if not isinstance(value, tuple):
            return base_offset

        tile_coord = value[0]
        victim_color = value[1] if len(value) > 1 else None

        # Get tile index
        tile_idx = self.tile_to_idx.get(tile_coord, 0)

        # Get victim index (0=none, 1=RED, 2=BLUE, 3=WHITE, 4=ORANGE)
        victim_idx = self._color_to_victim_idx(victim_color)

        # Encode: tile_idx * 5 + victim_idx
        return base_offset + tile_idx * 5 + victim_idx

    def _color_to_victim_idx(self, color) -> int:
        """Convert color to victim index (0 for none/self)."""
        if color is None:
            return 0
        color_order = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
        try:
            return 1 + color_order.index(color)
        except ValueError:
            return 0

    def _encode_maritime_trade(self, value) -> int:
        """Encode maritime trade action.

        Trades have 3 ratios:
        - 4:1 (standard): ('BRICK', 'BRICK', 'BRICK', 'BRICK', 'WOOD')
        - 3:1 (with 3:1 port): ('BRICK', 'BRICK', 'BRICK', None, 'WOOD')
        - 2:1 (with resource port): ('BRICK', 'BRICK', None, None, 'WOOD')

        Encoding: 5 give × 4 receive × 3 ratios = 60 actions
        """
        if not isinstance(value, tuple) or len(value) < 5:
            return self.MARITIME_TRADE_START

        # Determine ratio by counting non-None values
        non_none = sum(1 for v in value[:-1] if v is not None)
        if non_none == 4:
            ratio_idx = 0  # 4:1
        elif non_none == 3:
            ratio_idx = 1  # 3:1
        else:
            ratio_idx = 2  # 2:1

        # Determine what we're giving (first non-None value)
        give_resource = str(value[0])
        # Determine what we're receiving (last one)
        receive_resource = str(value[-1])

        give_idx = self._resource_to_idx(give_resource)
        receive_idx = self._resource_to_idx(receive_resource)

        if give_idx < 0 or receive_idx < 0 or give_idx == receive_idx:
            return self.MARITIME_TRADE_START

        # Adjust receive_idx to skip over give_idx (gives us 0-3 for each give)
        if receive_idx > give_idx:
            adjusted_receive = receive_idx - 1
        else:
            adjusted_receive = receive_idx

        # Encode: (ratio_idx * 5 * 4) + (give_idx * 4) + adjusted_receive
        return self.MARITIME_TRADE_START + ratio_idx * 20 + give_idx * 4 + adjusted_receive

    def _encode_discard(self, value) -> int:
        """Encode discard action (simplified to patterns)."""
        # Discard is complex, use hash-based encoding to 50 slots
        if value is None:
            return self.DISCARD_START

        # Hash the discard value
        h = hash(str(value))
        return self.DISCARD_START + (h % 50)

    def _resource_to_idx(self, resource) -> int:
        """Convert resource string to index 0-4."""
        res_str = str(resource).upper()
        # Handle enum-like strings
        if '.' in res_str:
            res_str = res_str.split('.')[-1]

        mapping = {'WOOD': 0, 'BRICK': 1, 'SHEEP': 2, 'WHEAT': 3, 'ORE': 4}
        return mapping.get(res_str, -1)

    def _resource_pair_idx(self, value) -> int:
        """Convert resource pair to index 0-14."""
        # 15 combinations: 5 same-same + 10 different pairs
        if not isinstance(value, tuple) or len(value) < 2:
            return 0

        r1 = self._resource_to_idx(value[0])
        r2 = self._resource_to_idx(value[1])

        if r1 < 0 or r2 < 0:
            return 0

        # Sort for consistency
        if r1 > r2:
            r1, r2 = r2, r1

        # Same resource: indices 0-4
        if r1 == r2:
            return r1

        # Different resources: indices 5-14
        # For pairs (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        idx = 5
        for i in range(5):
            for j in range(i + 1, 5):
                if i == r1 and j == r2:
                    return idx
                idx += 1

        return 5  # Fallback

    def _build_tile_mapping(self, catan_map):
        """Build tile coordinate to index mapping."""
        sorted_tiles = sorted(catan_map.land_tiles.keys())
        self.tile_list = sorted_tiles
        self.tile_to_idx = {coord: i for i, coord in enumerate(sorted_tiles)}

    def get_valid_action_mask(self, valid_actions, game: Game = None) -> np.ndarray:
        """Create a binary mask of valid actions.

        Args:
            valid_actions: List of valid Action objects
            game: The game for context (to build tile mapping)

        Returns:
            numpy array of shape (action_space_size,) with 1.0 for valid actions
        """
        mask = np.zeros(self.action_space_size, dtype=np.float32)

        for action in valid_actions:
            idx = self.encode(action, game)
            if 0 <= idx < self.action_space_size:
                mask[idx] = 1.0

        return mask


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_state_size() -> int:
    """Get the state encoding size."""
    return StateEncoder().state_size


def get_action_size() -> int:
    """Get the action space size."""
    return ActionEncoder().action_space_size


# Quick verification
if __name__ == "__main__":
    print(f"Number of edges in edge list: {NUM_EDGES}")
    print(f"State size: {get_state_size()}")
    print(f"Action size: {get_action_size()}")

    encoder = ActionEncoder()
    print("\nAction offsets:")
    print(f"  ROLL: {encoder.ROLL_START}")
    print(f"  END_TURN: {encoder.END_TURN_START}")
    print(f"  BUILD_SETTLEMENT: {encoder.BUILD_SETTLEMENT_START}-{encoder.BUILD_SETTLEMENT_START + NUM_NODES - 1}")
    print(f"  BUILD_CITY: {encoder.BUILD_CITY_START}-{encoder.BUILD_CITY_START + NUM_NODES - 1}")
    print(f"  BUILD_ROAD: {encoder.BUILD_ROAD_START}-{encoder.BUILD_ROAD_START + NUM_EDGES - 1}")
    print(f"  BUY_DEV_CARD: {encoder.BUY_DEV_CARD_START}")
    print(f"  PLAY_KNIGHT: {encoder.PLAY_KNIGHT_START}-{encoder.PLAY_KNIGHT_START + NUM_TILES * 5 - 1}")
    print(f"  MOVE_ROBBER: {encoder.MOVE_ROBBER_START}-{encoder.MOVE_ROBBER_START + NUM_TILES * 5 - 1}")
    print(f"  YEAR_OF_PLENTY: {encoder.YEAR_OF_PLENTY_START}-{encoder.YEAR_OF_PLENTY_START + 14}")
    print(f"  MONOPOLY: {encoder.MONOPOLY_START}-{encoder.MONOPOLY_START + 4}")
    print(f"  ROAD_BUILDING: {encoder.ROAD_BUILDING_START}")
    print(f"  MARITIME_TRADE: {encoder.MARITIME_TRADE_START}-{encoder.MARITIME_TRADE_START + 19}")
    print(f"  DISCARD: {encoder.DISCARD_START}-{encoder.DISCARD_START + 49}")
