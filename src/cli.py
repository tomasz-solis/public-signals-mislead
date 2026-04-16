"""Console-entry wrappers for repo commands exposed through ``pyproject.toml``."""


def apply_outcomes_main() -> None:
    """Run the decision-context labeling command."""
    from scripts.apply_outcomes import main

    main()


def generate_viz_main() -> None:
    """Run the visualization-generation command."""
    from scripts.generate_visualizations import main

    main()
