# scripts/generate_synthetic_preview_03.py: run the softened multi-holder preview_03 generator.

from generate_synthetic_labelme import main


if __name__ == "__main__":
    main(default_condition="preview_03", forced_geometry_profile="preview_03")
