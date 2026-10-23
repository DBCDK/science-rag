# Standard columns to keep for metadata/exclude from the CSV files
# For aktiviteter
AKTIVITETER_METADATA_COLS = [
    "ID",
    "Title",
    "URL",
    "[Manchet] Varighed",
    "[Manchet] Niveau",
    "Fag",
    "Klassetrin",
    "Emneord",
    "Kategori",
]
AKTIVITETER_EXCLUDE_COLS = ["Which tabs to show", "page_content_raw", "page_content"]
AKTIVITETER_EXCLUDE_COL_IF_CONTAINS = []

# For forløb
FORLOB_METADATA_COLS = ["ID", "Title", "URL", "Varighed", "Partnere", "Tilknyttede aktiviteter"]
FORLOB_EXCLUDE_COL_IF_CONTAINS = ["download_or_link", "pdf_link"]
FORLOB_EXCLUDE_COLS = [
    "Hvilke faner skal vises",
    "Video url",
    "Sidebar email_acf_education_material_sidebar_boxes_email_header",
    "Sidebar email_acf_education_material_sidebar_boxes_email_content",
    "page_content_raw",
    "page_content",
]

# For undersøgelser (testoteket.dk experiments). Permalink is renamed to URL when the csv is read
UNDERSOEGELSER_METADATA_COLS = ["ID", "Title", "URL", "[Manchet] Varighed", "[Manchet] Niveau"]
UNDERSOEGELSER_EXCLUDE_COL_IF_CONTAINS = ["Video url"]
UNDERSOEGELSER_EXCLUDE_COLS = [
    "Date",
    "Post Type",
    "Overskriv fra sheets",
    "Standardtekst",
    "Image URL",
    "Image Filename",
    "Image Path",
    "Image ID",
    "Image Title",
    "Image Featured",
    "page_content_raw",
    "page_content",
]
