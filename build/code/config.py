from pathlib import Path

# ————————————————————————————— #
#     Core Paths
# ————————————————————————————— #
CODE_PATH = Path.cwd()
PARENT_PATH = CODE_PATH.parent

# ————————————————————————————— #
#     Build Area
# ————————————————————————————— #
BUILD_ROOT = PARENT_PATH / "build"
BUILD_CODE = BUILD_ROOT / "code"
BUILD_DATA = BUILD_ROOT / "data"
BUILD_DATA_RAW = BUILD_DATA / "raw"
MACRO_DATA_PATH = BUILD_DATA_RAW / "Macro"
REVELIO_DATA_PATH = BUILD_DATA_RAW / "Revelio"
PITCHBOOK_DATA_PATH = BUILD_DATA_RAW / "Pitchbook"
AA_DATA_PATH = BUILD_DATA_RAW / "Audit Analytics"
AT_DATA_PATH = BUILD_DATA_RAW / "Accounting Today"
BUILD_DATA_TEMP = BUILD_DATA / "temp"
BUILD_DATA_PROCESSED = BUILD_DATA / "processed"

# ————————————————————————————— #
#     Analysis Area
# ————————————————————————————— #
ANALYSIS_ROOT = PARENT_PATH / "analysis"
ANALYSIS_CODE = ANALYSIS_ROOT / "code"
ANALYSIS_DATA = ANALYSIS_ROOT / "data"
ANALYSIS_INPUT = ANALYSIS_DATA / "input"
ANALYSIS_OUTPUT = ANALYSIS_DATA / "output"


# ————————————————————————————— #
#     Debug / Display
# ————————————————————————————— #
if __name__ == "__main__":
    print("|" + "-" * 78)
    print("|CODE_PATH:               ", CODE_PATH)
    print("|PARENT_PATH:             ", PARENT_PATH)
    print("|" + "-" * 78)

    print("|BUILD_ROOT:              ", BUILD_ROOT)
    print("|  BUILD_CODE:            ", BUILD_CODE)
    print("|  BUILD_DATA:            ", BUILD_DATA)
    print("|    RAW:                 ", BUILD_DATA_RAW)
    print("|      - Macro:           ", MACRO_DATA_PATH)
    print("|      - Revelio:         ", REVELIO_DATA_PATH)
    print("|      - Pitchbook:       ", PITCHBOOK_DATA_PATH)
    print("|      - Audit Analytics: ", AA_DATA_PATH)
    print("|      - Accounting Today:", AT_DATA_PATH)
    print("|    TEMP:                ", BUILD_DATA_TEMP)
    print("|    PROCESSED:           ", BUILD_DATA_PROCESSED)
    print("|" + "-" * 78)

    print("|ANALYSIS_ROOT:           ", ANALYSIS_ROOT)
    print("|  ANALYSIS_CODE:         ", ANALYSIS_CODE)
    print("|  ANALYSIS_DATA:         ", ANALYSIS_DATA)
    print("|    INPUT:               ", ANALYSIS_INPUT)
    print("|    Only for data that can be used for generating tables and figures.")
    print("|    OUTPUT:              ", ANALYSIS_OUTPUT)
    print("|    Only for tables and figures.")
    print("|" + "-" * 78)
