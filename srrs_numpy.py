import io
import numpy as np

# Read an SRRS file into memory
srrs_in = io.open('2018090100_SRRS0248.PCX', 'rb')
srrs = np.fromfile(srrs_in, dtype=np.dtype('|S1'))

# Locate all '*'s in the file (1's indicate a '*'; 0's indicate anything else. Then element-wise subtract the 1's and
#  0's to show begin and end positions for strings of '*'s (1's indicate begins; -1's indicate ends)
diffs = np.diff((srrs == '*').astype(np.int8, copy=False))
diffs = np.insert(diffs, 0, 1)

# Create arrays of array indices where the begins and ends are located. Subtract end indices from begin indices to get
# the lengths of the strings of stars.
starts = np.argwhere(diffs == 1)
stops = np.argwhere(diffs == -1)
num_stars = (stops - starts).flatten()

# Stack the begin and end indices next to one another (like tuples) for logical grouping. Filter out stars that aren't
# part of a 4 star group.
four_stars = np.column_stack((starts, stops))[(num_stars == 4)]

# Determine the distance between each group of 4 stars and its immediate neighbors (next and previous in the array).
next_dist = np.append(np.diff(four_stars[:, 0]), 0)
prev_dist = np.roll(next_dist, 1)

# Throw out any group of 4 stars that isn't within 14 bytes of another group of 4 stars (isn't paired).
four_star_pairs = four_stars[np.logical_or(next_dist == 14, prev_dist == 14)]

# Logically group pairs of 4 stars into tuple-like array. See below:
bulletins = four_star_pairs.reshape(-1, 4)

#  rec_start, len_start, len_end, star_end
# [        0          4       14       18]
# [     1626       1630     1640     1644]
# [     7396       7400     7410     7414]

# Stick index of next record as 5th column on to our tuple-like array. "Tuples" can now correctly delineate a
# valid bulletin from SRRS. See example below:
next_record_field = np.append(bulletins[1:, 0], srrs.size)
bulletins = np.column_stack((bulletins, next_record_field))

#  rec_start, len_start, len_end, star_end, next_rec
#  [       0          4       14        18     1626]
#  [    1626       1630     1640      1644     7396]
#  [    7396       7400     7410      7414     8388]


for rec_start, len_start, len_end, star_end, next_rec in bulletins:
    print(''.join(srrs[rec_start: star_end]), (next_rec - rec_start - 19))
