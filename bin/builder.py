# coding: utf-8 
'''
""" MIT License """
    Authors:
        1. Axel Masquelin
    Copyright (c) 2022
'''

# ---------------- Libaries --------------- #
from argparse import Namespace
import argparse
import yaml
import os
# ----------------------------------------- #

def build_yaml() -> argparse.ArgumentParser:
    """
    Parsers that takes the location of the configuration yaml document for the experiment parameters
    --------
    Returns:
        (1) parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to Yaml file for training/inference"
    )

    return parser


def build_config(args, logger):
    """
    -----------
    Parameters:
        (1) args: parser containing the location of the yaml file
    --------
    Returns:
        (1) data: config dictionary that contains the experiment parameters.
    """
    with open(os.getcwd() + args.config, "r") as yamlfile:
        try:
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            return data

        except Exception as e:
            logger.error(e, stack_info=True, exc_info=True)
