********************************************************************************
* Listing packages to check and install if missing
********************************************************************************
// 1. List all the packages you want
local pkgs spmap shp2dta mif2dta egenmore binscatter

// 2. Loop over each package, check with ssc describe, install if missing
foreach pkg of local pkgs {
    quietly ssc describe `pkg'      // _rc==0 if installed, ≠0 otherwise :contentReference[oaicite:0]{index=0}
    if _rc {
        display as text `"📦 Installing `pkg' from SSC..."'
        quietly ssc install `pkg', replace
    }
    else {
        display as result `"✔ `pkg' is already installed."'
    }
}

********************************************************************************
* sets.do - Cross-platform, portable path setup
********************************************************************************

* Save current working directory
global code "`c(pwd)'"


mata:
    // Get current code path from Stata global macro
    code = st_global("code")

    // Extract upper-level directories
    analysis = pathgetparent(code)
    parent = pathgetparent(analysis)

    // Save them back to Stata as global macros
    st_global("analysis", analysis)
    st_global("parent", parent)

    // Detect the path separator used in code
    pos_slash  = strpos(code, "/")
    pos_bslash = strpos(code, "\")

    if (pos_slash > 0 & (pos_bslash == 0 | pos_slash < pos_bslash)) {
        sep = "/"
    } else if (pos_bslash > 0) {
        sep = "\"
    } else {
        sep = "/"  // fallback to forward slash
    }

    // Save separator to global macro
    st_global("sep", sep)
end




********************************************************************************
* 📁 Main Project Folders
********************************************************************************

* ✨ Build Area
global BUILD_ROOT               "${PARENT}${SEP}build"
    global BUILD_CODE           "${BUILD_ROOT}${SEP}code"
    global BUILD_DATA           "${BUILD_ROOT}${SEP}data"
        global BUILD_DATA_RAW           "${BUILD_DATA}${SEP}raw"
        global BUILD_DATA_TEMP          "${BUILD_DATA}${SEP}temp"
        global BUILD_DATA_PROCESSED     "${BUILD_DATA}${SEP}processed"

* ✨ Analysis Area
global ANALYSIS_ROOT            "${PARENT}${SEP}analysis"
    global ANALYSIS_CODE        "${ANALYSIS_ROOT}${SEP}code"
    global ANALYSIS_DATA        "${ANALYSIS_ROOT}${SEP}data"
        global ANALYSIS_INPUT          "${ANALYSIS_DATA}${SEP}input"
        global ANALYSIS_OUTPUT         "${ANALYSIS_DATA}${SEP}output"


********************************************************************************
* 📁 Display Current Configuration
********************************************************************************

* ✨ Core Settings
display "--------------------------------------------"
display "📁 PARENT:                $PARENT"
display "📁 SEP:                   `$SEP'"
display "--------------------------------------------"

* ✨ Build Area
display "📁 BUILD_ROOT:            $BUILD_ROOT"
display "  📁 BUILD_CODE:          $BUILD_CODE"
display "  📁 BUILD_DATA:          $BUILD_DATA"
display "    📁 RAW:               $BUILD_DATA_RAW"
display "    📁 TEMP:              $BUILD_DATA_TEMP"
display "    📁 PROCESSED:         $BUILD_DATA_PROCESSED"
display "--------------------------------------------"

* ✨ Analysis Area
display "📁 ANALYSIS_ROOT:         $ANALYSIS_ROOT"
display "  📁 ANALYSIS_CODE:       $ANALYSIS_CODE"
display "  📁 ANALYSIS_DATA:       $ANALYSIS_DATA"
display "    📁 INPUT:             $ANALYSIS_INPUT"
display	"	 Only for data that can be used for generating tables and figures."
display "    📁 OUTPUT:            $ANALYSIS_OUTPUT"
display	"	 Only for tables and figures"
display "--------------------------------------------"


