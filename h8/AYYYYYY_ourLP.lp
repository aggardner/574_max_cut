\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  9 z0_1 + 10 z0_6 + 5 z0_7 + 11 z1_2 + 7 z1_4 + 3 z1_5 + 7 z1_6 + 2 z1_7
   + 10 z2_5 + 6 z2_6 + 2 z3_5 + 4 z3_6 + z3_7 + 8 z4_5 + 10 z4_7 + 8 z5_6
   + 6 z5_7 + 5 z6_7
Subject To
 partition_0: - x6 - x7 + z6_7 <= 0
 fancy_0: x6 + x7 + z6_7 <= 2
 partition_1: - x3 - x5 + z3_5 <= 0
 fancy_1: x3 + x5 + z3_5 <= 2
 partition_2: - x3 - x7 + z3_7 <= 0
 fancy_2: x3 + x7 + z3_7 <= 2
 partition_3: - x3 - x6 + z3_6 <= 0
 fancy_3: x3 + x6 + z3_6 <= 2
 partition_4: - x2 - x5 + z2_5 <= 0
 fancy_4: x2 + x5 + z2_5 <= 2
 partition_5: - x2 - x6 + z2_6 <= 0
 fancy_5: x2 + x6 + z2_6 <= 2
 partition_6: - x1 - x7 + z1_7 <= 0
 fancy_6: x1 + x7 + z1_7 <= 2
 partition_7: - x1 - x6 + z1_6 <= 0
 fancy_7: x1 + x6 + z1_6 <= 2
 partition_8: - x1 - x5 + z1_5 <= 0
 fancy_8: x1 + x5 + z1_5 <= 2
 partition_9: - x1 - x4 + z1_4 <= 0
 fancy_9: x1 + x4 + z1_4 <= 2
 partition_10: - x5 - x7 + z5_7 <= 0
 fancy_10: x5 + x7 + z5_7 <= 2
 partition_11: - x1 - x2 + z1_2 <= 0
 fancy_11: x1 + x2 + z1_2 <= 2
 partition_12: - x0 - x6 + z0_6 <= 0
 fancy_12: x0 + x6 + z0_6 <= 2
 partition_13: - x0 - x7 + z0_7 <= 0
 fancy_13: x0 + x7 + z0_7 <= 2
 partition_14: - x4 - x5 + z4_5 <= 0
 fancy_14: x4 + x5 + z4_5 <= 2
 partition_15: - x4 - x7 + z4_7 <= 0
 fancy_15: x4 + x7 + z4_7 <= 2
 partition_16: - x5 - x6 + z5_6 <= 0
 fancy_16: x5 + x6 + z5_6 <= 2
 partition_17: - x0 - x1 + z0_1 <= 0
 fancy_17: x0 + x1 + z0_1 <= 2
Bounds
Binaries
 x0 x1 x2 x3 x4 x5 x6 x7 z0_1 z0_6 z0_7 z1_2 z1_4 z1_5 z1_6 z1_7 z2_5 z2_6
 z3_5 z3_6 z3_7 z4_5 z4_7 z5_6 z5_7 z6_7
End
