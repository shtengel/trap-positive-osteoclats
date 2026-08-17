import re

def _natural_sort_key(name):
    """Return a sort key that treats consecutive digits as numbers."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', name)
    ]

def sort_images_by_group_and_column(images, groups=[("B", "C", "D"), ("E", "F", "G")]):
    # Build group priority map: 'B' → (0, 0), 'C' → (0, 1), etc.
    group_priority = {
        letter: (group_idx, letter_idx)
        for group_idx, group in enumerate(groups)
        for letter_idx, letter in enumerate(group)
    }

    def parse_image(obj):
        name = obj["image_name"]
        match = re.search(r'_([A-Z])(\d{2})f', name)
        if match:
            row_letter = match.group(1)
            col_number = int(match.group(2))
            return row_letter, col_number
        return None, None  # Fallback for bad format

    def sort_key(obj):
        name = obj["image_name"]
        match = re.search(r'_([A-Z])(\d{2})f', name)
        if match:
            row_letter = match.group(1)
            col_number = int(match.group(2))
            group_info = group_priority.get(row_letter, (float('inf'), float('inf')))
            return (col_number, group_info)
        return (float('inf'), (float('inf'), float('inf')))  # fallback for bad format

    result = []
    for group in groups:
        # Filter images in current group
        group_images = []
        for image in images:
            row_letter, col_number = parse_image(image)
            if row_letter in group:
                group_images.append(image)

        # Sort by col_number
        group_images = sorted(group_images, key=sort_key)

        # Append sorted images
        result = result + group_images

    # Append any files whose name didn't match the expected pattern
    # (e.g. `_[A-Z]\d{2}f`). Sorted naturally so numeric filenames order as 1, 2, 3, ..., 10.
    matched_names = {img["image_name"] for img in result}
    unmatched = [img for img in images if img["image_name"] not in matched_names]
    result = result + sorted(unmatched, key=lambda img: _natural_sort_key(img["image_name"]))

    return result