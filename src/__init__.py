"""
Source code for public-signals-mislead analysis.

This package contains the code for analyzing how public signals can mislead
product decisions when they are used without internal context.

The repo deliberately separates observable company action from harder-to-observe
business value. Public data can show how attention and sentiment move. It often
cannot show whether a feature truly helped the business or a specific audience.

Modules:
    - data_collection: Scripts for collecting Google Trends and Reddit data
    - analysis: Statistical tests and decision-support summaries
    - visualization: Plotly charts and interactive visualizations
"""

__all__ = ["data_collection", "analysis", "visualization"]
