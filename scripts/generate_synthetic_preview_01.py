# scripts/generate_synthetic_preview_01.py: run the large near-top-view preview_01 generator.

from generate_synthetic_labelme import main


if __name__ == "__main__":
    main(default_condition="preview_01", forced_geometry_profile="preview_01")
