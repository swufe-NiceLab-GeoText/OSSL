# OSSL

## Framework of OSSL

 <img alt="Figure 2: The framework of OSSL" height="400" src="images/framework.png" width="400"/>
 
## Case Study of OSSL

 <img alt="Figure 8: A case study on ATDO task with OSSL" height="400" src="images/case_study.png" width="400"/>

Ultimately, we conduct a case study to intuitively illustrate our OSSL’s performance on the ATDO task under the OW context. We randomly select an SD pair in the Porto dataset and visualize its corresponding mapped trajectories, as depicted in Fig 8, The top is the ground truth, reflecting the trajectory with real patterns, along the bottom shows the recognition results from our OSSL. The four sub-figures from left to right show the normal class (c1, known class), detour (c2, known class), unknown 1 (actually route-switch), and unknown 2 (actually navigation). First, we can see that our OSSL demonstrates a distinct impact on the differentiation of known and unknown classes, successfully distinguishing seven trajectories of all classes under our OW settings. Besides, there are two misclassifications. The intriguing as $T_{1559}$, the unknown 1 (Ground Truth) vs. detour (prediction), which shares traits with detours from a human perspective, such as an increased distance from the S to D during transition. Another case, $T_{1798}$, is misclassified into unknown 1, despite belonging to unknown 2 in the settings. We conjecture that this may stem from the reported navigation being close to another route, resulting in ambiguous decision-making, hence the misclassification. Furthermore, our OSSL demonstrates its capability to learn both spatial and travel semantics of trajectories. For example, instances $T_{583}$ and $T_{102}$ are accurately classified as belonging to the normal class, reflecting their spatial semantic patterns.

