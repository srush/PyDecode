\* Hypergraph Problem *\
Minimize
OBJ: 0.865309927772 edge_0 + 0.805027827013 edge_1 + 0.719704686404 edge_2
 + 0.824844977148 edge_3 + 0.668153201232 edge_4 + 0.932833824227 edge_5
 + 0.551267246091 edge_6
Subject To
_C1: node_13_tri_right_0_3 = 1
_C10: - edge_0 + node_1_tri_right_1_1 = 0
_C11: - edge_1 + node_2_tri_left_1_1 = 0
_C12: - edge_2 + node_3_tri_right_2_2 = 0
_C13: - edge_0 + node_4_tri_left_2_2 = 0
_C14: - edge_3 + node_5_tri_right_3_3 = 0
_C15: - edge_2 + node_6_tri_left_3_3 = 0
_C16: - edge_1 + node_7_trap_left_1_2 = 0
_C17: - edge_4 + node_8_tri_left_1_2 = 0
_C18: - edge_3 + node_9_trap_right_2_3 = 0
_C19: - edge_5 + node_10_tri_right_2_3 = 0
_C2: - edge_0 + node_7_trap_left_1_2 = 0
_C20: - edge_5 + node_11_trap_right_0_2 = 0
_C21: - edge_6 + node_12_tri_right_0_3 = 0
_C3: - edge_1 + node_8_tri_left_1_2 = 0
_C4: - edge_2 + node_9_trap_right_2_3 = 0
_C5: - edge_3 + node_10_tri_right_2_3 = 0
_C6: - edge_4 + node_11_trap_right_0_2 = 0
_C7: - edge_5 + node_12_tri_right_0_3 = 0
_C8: - edge_6 + node_13_tri_right_0_3 = 0
_C9: - edge_4 + node_0_tri_right_0_0 = 0
Bounds
edge_0 <= 1
edge_1 <= 1
edge_2 <= 1
edge_3 <= 1
edge_4 <= 1
edge_5 <= 1
edge_6 <= 1
node_0_tri_right_0_0 <= 1
node_10_tri_right_2_3 <= 1
node_11_trap_right_0_2 <= 1
node_12_tri_right_0_3 <= 1
node_13_tri_right_0_3 <= 1
node_1_tri_right_1_1 <= 1
node_2_tri_left_1_1 <= 1
node_3_tri_right_2_2 <= 1
node_4_tri_left_2_2 <= 1
node_5_tri_right_3_3 <= 1
node_6_tri_left_3_3 <= 1
node_7_trap_left_1_2 <= 1
node_8_tri_left_1_2 <= 1
node_9_trap_right_2_3 <= 1
End
