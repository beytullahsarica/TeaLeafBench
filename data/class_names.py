"""
Display names of the classes. Folder / file names keep the internal snake_case names (stable, safe in paths);
everything people read -- figures, reports, LaTeX tables, the paper -- uses these names.
Change a name here only: figures, reports and tables all follow.
"""
DISPLAY = {
    'algal_leaf_spot': 'Algal Leaf Spot',
    'anthracnose': 'Anthracnose',
    'bird_eye_spot': "Bird's Eye Spot",
    'black_rot': 'Black Rot',
    'brown_blight': 'Brown Blight',
    'gray_blight': 'Grey Blight',            # British spelling, as in the paper
    'green_mirid_bug': 'Green Mirid Bug',
    'healthy': 'Healthy',
    'helopeltis': 'Helopeltis',               # genus name -> italic (see display_name)
    'leaf_rust': 'Leaf Rust',
    'red_leaf_spot': 'Red Leaf Spot',
    'red_spider': 'Red Spider Mite',
    'tea_mosquito_bug': 'Tea Mosquito Bug',
    'white_spot': 'White Spot',
}
ITALIC = {'helopeltis'}                       # scientific (genus) names


def display_name(c, fmt='plain'):
    """fmt: 'plain' (reports, CSV), 'mpl' (matplotlib labels, mathtext italic), 'tex' (LaTeX)."""
    name = DISPLAY.get(c, c.replace('_', ' ').title())
    if c in ITALIC:
        if fmt == 'mpl':
            return r'$\it{' + name.replace(' ', r'\ ') + '}$'
        if fmt == 'tex':
            return r'\emph{' + name + '}'
    return name


def display_names(classes, fmt='plain'):
    return [display_name(c, fmt) for c in classes]
