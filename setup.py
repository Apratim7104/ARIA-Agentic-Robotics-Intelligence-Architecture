"""Package setup for ARIA: Autonomous Robotic Manipulation System."""

from setuptools import find_packages, setup

setup(
    name="aria_manipulation",
    version="3.0.0",
    description="ARIA: Autonomous Robotic Manipulation Stack with Multi-Agent Planning, Spatial Reasoning, CCDM Dynamics, and Franka 7-DOF NMPC",
    author="ARIA Team",
    python_requires=">=3.10",
    packages=find_packages(include=[
        "multi_agent_orchestrator", "multi_agent_orchestrator.*",
        "spatial_layer", "spatial_layer.*",
        "ccdm_dynamics", "ccdm_dynamics.*",
        "ccdm", "ccdm.*"
    ]),
    install_requires=[
        "mujoco>=3.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0",
        "torch>=2.0.0",
        "casadi>=3.6.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.7.0",
        "reportlab>=4.0.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.23.0"],
        "llm": ["langgraph>=0.2.0", "langchain>=0.2.0", "openai>=1.30.0", "tavily-python>=0.3.0"],
    },
)
