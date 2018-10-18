#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

__arthur__ = "张俊鹏"

LATITUDE_RANGES = zip(range(32, 44), range(33, 45))

for r in LATITUDE_RANGES:
    print(r)
    print(r[0])
    print(r[1])

r = np.arange(1.0, 3.0)
print(r)
