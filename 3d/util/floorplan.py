import random
from PIL import Image, ImageDraw

# ============================================================
# CONFIGURATION
# ============================================================
IMAGE_WIDTH = 220
IMAGE_HEIGHT = 220
OUTPUT_FILE = "barrio_floorplan.png"

WALL_THICKNESS = 1
MARGIN = 2

# Barrio layout
BLOCK_COLS = 3
BLOCK_ROWS = 3
STREET_W = 12              # magenta asphalt roads between blocks
PLOT_PADDING = 3           # margin inside each block
TERRAIN_PADDING = 6        # green terrain ring around each house

# House generation
MIN_ROOM_W = 10
MIN_ROOM_H = 10
MAX_SPLIT_DEPTH = 3

# Crosswalk
CROSSWALK_STRIPE = 2
CROSSWALK_GAP = 2
CROSSWALK_MARGIN = 2

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 180, 0)        # walkable terrain / paths
BLUE = (0, 0, 255)         # beds
YELLOW = (255, 255, 0)     # living room resting places
RED = (255, 0, 0)          # WC
CYAN = (0, 255, 255)       # kitchen furniture
MAGENTA = (255, 0, 255)    # asphalt roads

# ============================================================
# DATA STRUCTURE
# ============================================================
class Room:
    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.kind = None

# ============================================================
# DRAW HELPERS
# ============================================================
def fill_rect(draw, x, y, w, h, color):
    if w <= 0 or h <= 0:
        return
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=color)

def draw_rect_outline(draw, x, y, w, h, color, width=1):
    if w <= 0 or h <= 0:
        return
    draw.rectangle([x, y, x + w - 1, y + h - 1], outline=color, width=width)

def fill_room_floor(draw, room, color):
    pad = WALL_THICKNESS
    x1 = room.x + pad
    y1 = room.y + pad
    x2 = room.x + room.w - pad - 1
    y2 = room.y + room.h - pad - 1
    if x2 >= x1 and y2 >= y1:
        draw.rectangle([x1, y1, x2, y2], fill=color)

def draw_room_walls(draw, room):
    draw_rect_outline(draw, room.x, room.y, room.w, room.h, WHITE, WALL_THICKNESS)

# ============================================================
# ROOM SPLITTING
# ============================================================
def split_room(room, depth=0):
    if depth >= MAX_SPLIT_DEPTH:
        return [room]

    can_split_h = room.h >= 2 * MIN_ROOM_H
    can_split_v = room.w >= 2 * MIN_ROOM_W

    if not can_split_h and not can_split_v:
        return [room]

    if depth > 1 and random.random() < 0.20:
        return [room]

    if can_split_h and can_split_v:
        split_vertical = random.choice([True, False])
    elif can_split_v:
        split_vertical = True
    else:
        split_vertical = False

    if split_vertical:
        cut = random.randint(MIN_ROOM_W, room.w - MIN_ROOM_W)
        r1 = Room(room.x, room.y, cut, room.h)
        r2 = Room(room.x + cut, room.y, room.w - cut, room.h)
    else:
        cut = random.randint(MIN_ROOM_H, room.h - MIN_ROOM_H)
        r1 = Room(room.x, room.y, room.w, cut)
        r2 = Room(room.x, room.y + cut, room.w, room.h - cut)

    return split_room(r1, depth + 1) + split_room(r2, depth + 1)

def assign_room_types(rooms):
    types = ["living", "kitchen", "wc"]
    while len(types) < len(rooms):
        types.append("bedroom")
    random.shuffle(types)
    for room, kind in zip(rooms, types):
        room.kind = kind

# ============================================================
# INTERIOR DOORS
# ============================================================
def carve_interior_doors(draw, rooms):
    door_len = 4

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = rooms[i]
            b = rooms[j]

            # Shared vertical wall
            if a.x + a.w == b.x or b.x + b.w == a.x:
                shared_top = max(a.y, b.y)
                shared_bottom = min(a.y + a.h, b.y + b.h)
                shared_len = shared_bottom - shared_top

                if shared_len >= 8:
                    door_y = random.randint(shared_top + 2, shared_bottom - door_len - 1)
                    wall_x = a.x + a.w if a.x + a.w == b.x else b.x + b.w
                    draw.line([wall_x, door_y, wall_x, door_y + door_len], fill=GREEN, width=WALL_THICKNESS)

            # Shared horizontal wall
            if a.y + a.h == b.y or b.y + b.h == a.y:
                shared_left = max(a.x, b.x)
                shared_right = min(a.x + a.w, b.x + b.w)
                shared_len = shared_right - shared_left

                if shared_len >= 8:
                    door_x = random.randint(shared_left + 2, shared_right - door_len - 1)
                    wall_y = a.y + a.h if a.y + a.h == b.y else b.y + b.h
                    draw.line([door_x, wall_y, door_x + door_len, wall_y], fill=GREEN, width=WALL_THICKNESS)

# ============================================================
# FURNITURE HELPERS
# ============================================================
def random_rect_inside(room, min_w, min_h, max_w, max_h, margin=2):
    inner_x1 = room.x + WALL_THICKNESS + margin
    inner_y1 = room.y + WALL_THICKNESS + margin
    inner_x2 = room.x + room.w - WALL_THICKNESS - margin - 1
    inner_y2 = room.y + room.h - WALL_THICKNESS - margin - 1

    available_w = inner_x2 - inner_x1 + 1
    available_h = inner_y2 - inner_y1 + 1

    if available_w < min_w or available_h < min_h:
        return None

    w = random.randint(min_w, min(max_w, available_w))
    h = random.randint(min_h, min(max_h, available_h))
    x = random.randint(inner_x1, inner_x2 - w + 1)
    y = random.randint(inner_y1, inner_y2 - h + 1)

    return (x, y, w, h)

# ============================================================
# FURNITURE
# ============================================================
def draw_beds(draw, room):
    count = random.randint(1, 2)
    for _ in range(count):
        rect = random_rect_inside(room, 4, 3, 8, 5, margin=2)
        if rect:
            x, y, w, h = rect
            fill_rect(draw, x, y, w, h, BLUE)

def draw_kitchen_furniture(draw, room):
    count = random.randint(2, 4)
    for _ in range(count):
        rect = random_rect_inside(room, 2, 2, 5, 3, margin=2)
        if rect:
            x, y, w, h = rect
            fill_rect(draw, x, y, w, h, CYAN)

def draw_living_rest_places(draw, room):
    count = random.randint(1, 3)
    for _ in range(count):
        rect = random_rect_inside(room, 4, 2, 8, 4, margin=2)
        if rect:
            x, y, w, h = rect
            fill_rect(draw, x, y, w, h, YELLOW)

def draw_wc(draw, room):
    rect = random_rect_inside(room, 4, 4, 8, 8, margin=2)
    if rect:
        x, y, w, h = rect
        fill_rect(draw, x, y, w, h, RED)

# ============================================================
# HOUSE OUTER DOORS
# ============================================================
def carve_house_door(draw, house_x, house_y, house_w, house_h, side, door_center, door_len=5):
    half = door_len // 2

    if side == "top":
        x1 = max(house_x + 1, door_center - half)
        x2 = min(house_x + house_w - 2, door_center + half)
        draw.line([x1, house_y, x2, house_y], fill=GREEN, width=WALL_THICKNESS)
        return (door_center, house_y)

    if side == "bottom":
        y = house_y + house_h - 1
        x1 = max(house_x + 1, door_center - half)
        x2 = min(house_x + house_w - 2, door_center + half)
        draw.line([x1, y, x2, y], fill=GREEN, width=WALL_THICKNESS)
        return (door_center, y)

    if side == "left":
        y1 = max(house_y + 1, door_center - half)
        y2 = min(house_y + house_h - 2, door_center + half)
        draw.line([house_x, y1, house_x, y2], fill=GREEN, width=WALL_THICKNESS)
        return (house_x, door_center)

    if side == "right":
        x = house_x + house_w - 1
        y1 = max(house_y + 1, door_center - half)
        y2 = min(house_y + house_h - 2, door_center + half)
        draw.line([x, y1, x, y2], fill=GREEN, width=WALL_THICKNESS)
        return (x, door_center)

# ============================================================
# PATHS
# ============================================================
def draw_walkway(draw, x1, y1, x2, y2, width=3):
    hw = width // 2

    if random.random() < 0.5:
        draw.rectangle([min(x1, x2), y1 - hw, max(x1, x2), y1 + hw], fill=GREEN)
        draw.rectangle([x2 - hw, min(y1, y2), x2 + hw, max(y1, y2)], fill=GREEN)
    else:
        draw.rectangle([x1 - hw, min(y1, y2), x1 + hw, max(y1, y2)], fill=GREEN)
        draw.rectangle([min(x1, x2), y2 - hw, max(x1, x2), y2 + hw], fill=GREEN)

# ============================================================
# CROSSWALKS
# ============================================================
def draw_crosswalk_vertical(draw, road_x, y_center, road_w, sidewalk_left_x, sidewalk_right_x):
    # Crosses a vertical road, so stripes are vertical thin rectangles across its width direction
    usable_y1 = y_center - 4
    usable_y2 = y_center + 4

    x = road_x + CROSSWALK_MARGIN
    while x < road_x + road_w - CROSSWALK_MARGIN:
        stripe_w = min(CROSSWALK_STRIPE, road_x + road_w - CROSSWALK_MARGIN - x)
        fill_rect(draw, x, usable_y1, stripe_w, usable_y2 - usable_y1 + 1, WHITE)
        x += CROSSWALK_STRIPE + CROSSWALK_GAP

    # connect green sidewalks right up to the road edges
    fill_rect(draw, sidewalk_left_x, y_center - 1, road_x - sidewalk_left_x, 3, GREEN)
    fill_rect(draw, road_x + road_w, y_center - 1, sidewalk_right_x - (road_x + road_w), 3, GREEN)

def draw_crosswalk_horizontal(draw, x_center, road_y, road_h, sidewalk_top_y, sidewalk_bottom_y):
    usable_x1 = x_center - 4
    usable_x2 = x_center + 4

    y = road_y + CROSSWALK_MARGIN
    while y < road_y + road_h - CROSSWALK_MARGIN:
        stripe_h = min(CROSSWALK_STRIPE, road_y + road_h - CROSSWALK_MARGIN - y)
        fill_rect(draw, usable_x1, y, usable_x2 - usable_x1 + 1, stripe_h, WHITE)
        y += CROSSWALK_STRIPE + CROSSWALK_GAP

    fill_rect(draw, x_center - 1, sidewalk_top_y, 3, road_y - sidewalk_top_y, GREEN)
    fill_rect(draw, x_center - 1, road_y + road_h, 3, sidewalk_bottom_y - (road_y + road_h), GREEN)

# ============================================================
# SINGLE PLOT / HOUSE
# ============================================================
def generate_house_in_plot(draw, plot_x, plot_y, plot_w, plot_h):
    terrain_x = plot_x + PLOT_PADDING
    terrain_y = plot_y + PLOT_PADDING
    terrain_w = plot_w - 2 * PLOT_PADDING
    terrain_h = plot_h - 2 * PLOT_PADDING

    house_x = terrain_x + TERRAIN_PADDING
    house_y = terrain_y + TERRAIN_PADDING
    house_w = terrain_w - 2 * TERRAIN_PADDING
    house_h = terrain_h - 2 * TERRAIN_PADDING

    if house_w < MIN_ROOM_W or house_h < MIN_ROOM_H:
        return None

    # terrain
    fill_rect(draw, terrain_x, terrain_y, terrain_w, terrain_h, GREEN)
    draw_rect_outline(draw, terrain_x, terrain_y, terrain_w, terrain_h, WHITE, WALL_THICKNESS)

    # house
    root = Room(house_x, house_y, house_w, house_h)
    rooms = split_room(root)
    assign_room_types(rooms)

    for room in rooms:
        fill_room_floor(draw, room, GREEN)

    for room in rooms:
        if room.kind == "bedroom":
            draw_beds(draw, room)
        elif room.kind == "kitchen":
            draw_kitchen_furniture(draw, room)
        elif room.kind == "living":
            draw_living_rest_places(draw, room)
        elif room.kind == "wc":
            draw_wc(draw, room)

    draw_rect_outline(draw, house_x, house_y, house_w, house_h, WHITE, WALL_THICKNESS)
    for room in rooms:
        draw_room_walls(draw, room)

    carve_interior_doors(draw, rooms)

    # choose a side facing the closest street edge of the plot
    side_options = []
    if plot_y > MARGIN:
        side_options.append("top")
    if plot_y + plot_h < IMAGE_HEIGHT - MARGIN:
        side_options.append("bottom")
    if plot_x > MARGIN:
        side_options.append("left")
    if plot_x + plot_w < IMAGE_WIDTH - MARGIN:
        side_options.append("right")

    if not side_options:
        side_options = ["top", "bottom", "left", "right"]

    side = random.choice(side_options)

    if side in ("top", "bottom"):
        door_center = random.randint(house_x + 3, house_x + house_w - 4)
    else:
        door_center = random.randint(house_y + 3, house_y + house_h - 4)

    door_pos = carve_house_door(draw, house_x, house_y, house_w, house_h, side, door_center, door_len=5)

    # walkway from house door to terrain wall / block edge
    if side == "top":
        target = (door_pos[0], terrain_y)
    elif side == "bottom":
        target = (door_pos[0], terrain_y + terrain_h - 1)
    elif side == "left":
        target = (terrain_x, door_pos[1])
    else:
        target = (terrain_x + terrain_w - 1, door_pos[1])

    draw_walkway(draw, door_pos[0], door_pos[1], target[0], target[1], width=3)

    # open terrain wall at that point
    if side == "top":
        draw.line([target[0] - 2, terrain_y, target[0] + 2, terrain_y], fill=GREEN, width=WALL_THICKNESS)
    elif side == "bottom":
        yy = terrain_y + terrain_h - 1
        draw.line([target[0] - 2, yy, target[0] + 2, yy], fill=GREEN, width=WALL_THICKNESS)
    elif side == "left":
        draw.line([terrain_x, target[1] - 2, terrain_x, target[1] + 2], fill=GREEN, width=WALL_THICKNESS)
    else:
        xx = terrain_x + terrain_w - 1
        draw.line([xx, target[1] - 2, xx, target[1] + 2], fill=GREEN, width=WALL_THICKNESS)

    return {
        "plot": (plot_x, plot_y, plot_w, plot_h),
        "terrain": (terrain_x, terrain_y, terrain_w, terrain_h),
        "house": (house_x, house_y, house_w, house_h),
        "exit_side": side,
        "exit_pos": target,
        "rooms": len(rooms)
    }

# ============================================================
# BARRIO LAYOUT
# ============================================================
def compute_barrio_grid():
    usable_w = IMAGE_WIDTH - 2 * MARGIN
    usable_h = IMAGE_HEIGHT - 2 * MARGIN

    total_street_w = (BLOCK_COLS - 1) * STREET_W
    total_street_h = (BLOCK_ROWS - 1) * STREET_W

    block_w = (usable_w - total_street_w) // BLOCK_COLS
    block_h = (usable_h - total_street_h) // BLOCK_ROWS

    if block_w < 2 * (PLOT_PADDING + TERRAIN_PADDING) + MIN_ROOM_W:
        raise ValueError("Image too small for current horizontal barrio configuration.")
    if block_h < 2 * (PLOT_PADDING + TERRAIN_PADDING) + MIN_ROOM_H:
        raise ValueError("Image too small for current vertical barrio configuration.")

    blocks = []
    for row in range(BLOCK_ROWS):
        for col in range(BLOCK_COLS):
            x = MARGIN + col * (block_w + STREET_W)
            y = MARGIN + row * (block_h + STREET_W)
            blocks.append((col, row, x, y, block_w, block_h))

    return blocks, block_w, block_h

# ============================================================
# MAIN GENERATION
# ============================================================
def generate_barrio():
    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BLACK)
    draw = ImageDraw.Draw(img)

    # base asphalt everywhere inside margin
    fill_rect(draw, MARGIN, MARGIN, IMAGE_WIDTH - 2 * MARGIN, IMAGE_HEIGHT - 2 * MARGIN, MAGENTA)

    blocks, block_w, block_h = compute_barrio_grid()

    houses = []
    for col, row, x, y, w, h in blocks:
        info = generate_house_in_plot(draw, x, y, w, h)
        if info:
            houses.append(info)

    # vertical crosswalks between left-right neighboring blocks
    for row in range(BLOCK_ROWS):
        for col in range(BLOCK_COLS - 1):
            left_block_x = MARGIN + col * (block_w + STREET_W)
            right_block_x = MARGIN + (col + 1) * (block_w + STREET_W)
            road_x = left_block_x + block_w

            left_plot_center_y = MARGIN + row * (block_h + STREET_W) + block_h // 2
            draw_crosswalk_vertical(
                draw,
                road_x,
                left_plot_center_y,
                STREET_W,
                left_block_x + block_w - PLOT_PADDING,
                right_block_x + PLOT_PADDING
            )

    # horizontal crosswalks between top-bottom neighboring blocks
    for row in range(BLOCK_ROWS - 1):
        for col in range(BLOCK_COLS):
            top_block_y = MARGIN + row * (block_h + STREET_W)
            bottom_block_y = MARGIN + (row + 1) * (block_h + STREET_W)
            road_y = top_block_y + block_h

            top_plot_center_x = MARGIN + col * (block_w + STREET_W) + block_w // 2
            draw_crosswalk_horizontal(
                draw,
                top_plot_center_x,
                road_y,
                STREET_W,
                top_block_y + block_h - PLOT_PADDING,
                bottom_block_y + PLOT_PADDING
            )

    img.save(OUTPUT_FILE)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Overall image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Barrio blocks: {BLOCK_COLS}x{BLOCK_ROWS}")
    print(f"Block size: {block_w}x{block_h}")
    print(f"Street width: {STREET_W}")
    print(f"Houses generated: {len(houses)}")

    for i, h in enumerate(houses, start=1):
        px, py, pw, ph = h["plot"]
        hx, hy, hw, hh = h["house"]
        print(
            f"{i}. plot=({px},{py},{pw},{ph}) "
            f"house=({hx},{hy},{hw},{hh}) "
            f"rooms={h['rooms']} exit={h['exit_side']}"
        )

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    generate_barrio()
