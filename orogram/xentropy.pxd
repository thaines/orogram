# Copyright 2022 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

cpdef double section_crossentropy(float p0, float p1, float q0, float q1, double log_q0, double log_q1) nogil
