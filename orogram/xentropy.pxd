# Copyright 2022 Tom SF Haines

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

cdef float section_crossentropy(float p0, float p1, float q0, float q1, float log_q0, float log_q1) nogil
