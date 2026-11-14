# MdocToTlt.py

Generates AreTomo3 `_TLT.txt` files from SerialEM `.mdoc` files.

## Requirements

Python 3 (standard library only; no extra packages needed).

## Usage

```
python3 MdocToTlt.py -i <input> -o <outdir> [--serial 0|1]
```

| Option | Description |
|---|---|
| `-i, --input` | `--serial 0`: full name of one `.mdoc` file.<br>`--serial 1`: prefix of the mdoc file names, including the directory. |
| `-o, --outdir` | Directory where the `_TLT.txt` files are saved. Created if it does not exist. |
| `--serial` | `0` (default): process a single mdoc file.<br>`1`: batch-process every `<prefix>*.mdoc` file. |

### Examples

Single file:
```
python3 MdocToTlt.py --serial 0 -i /data/mdoc/Position_120.mdoc -o /data/tlt
```
Writes `/data/tlt/Position_120_TLT.txt`.

Batch:
```
python3 MdocToTlt.py --serial 1 -i /data/mdoc/Position_ -o /data/tlt
```
Processes every `/data/mdoc/Position_*.mdoc` and writes one `_TLT.txt` per mdoc file.
If `-i` is a directory, the script processes every `*.mdoc` file in that directory.

**Note:** the prefix is a plain "starts with" match. For example, `-i /data/mdoc/Position_12`
matches `Position_12.mdoc` and `Position_12_2.mdoc`, but also `Position_120.mdoc`, `Position_121_3.mdoc`, and so on.

## Output format

The output file name is the mdoc file name with `.mdoc` replaced by `_TLT.txt`.
Each row corresponds to one `[ZValue = N]` section of the mdoc file. There are three columns:

| Column | Content |
|---|---|
| 1 | `TiltAngle`. Rows are sorted by this column in ascending order. |
| 2 | 1-based acquisition index from `DateTime`. Index 1 is the earliest image. |
| 3 | `ExposureDose` |

Rows are written with the format `%8.2f%6d%10.2f`, for example:
```
  -45.05    31      1.54
  -42.05    28      1.59
   ...
   -0.05     1      1.94
    2.95     2      1.94
```

## Behavior notes

- Header lines before the first `[ZValue]` section, including `[T = ...]` lines, are ignored.
- If two sections have the same `DateTime`, the section with the lower ZValue gets the earlier index.
- If `DateTime` is missing or cannot be parsed in any section, the acquisition order falls back to ZValue order and the script prints a warning.
- A missing `ExposureDose` is written as `0.00`. A missing `TiltAngle` is an error for that file.
- In batch mode, a file that fails is reported and skipped, and the script continues with the remaining files.
  At the end it prints how many files were processed and how many failed.
- Exit code is `1` if any file failed, if no mdoc files matched, or if `--serial 0` is given
  a name that is not an existing `.mdoc` file.
