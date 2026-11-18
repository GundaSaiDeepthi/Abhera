"""
Module for Chapter 14: Complete Viva Preparation (Assembles Part 1 and Part 2)
"""

from ch14_viva_part1 import get_viva_part1
from ch14_viva_part2 import get_viva_part2

def get_chapter_14():
    return get_viva_part1() + "\n\n" + get_viva_part2()
