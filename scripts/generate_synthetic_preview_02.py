# scripts/generate_synthetic_preview_02.py: run the smaller tilted trapezoid preview_02 generator.

from generate_synthetic_labelme import main


if __name__ == "__main__":
    main(default_condition="preview_02", forced_geometry_profile="preview_02")
