
Author -- Yashwanth R  @https://github.com/ShiroganeMiyuki-0

Summary of NeruIPS-2021 research papers:
- Machine unlearning: The task of updating a machine learning (ML) model after some of the training data are deleted, so that the model reflects the remaining data.
- Desirable qualities: The ML model should be efficient, effective, and certifiable. Efficiency means low running time, effectiveness means high accuracy, and certifiability means guaranteeing that the deleted data are unlearned by the model.
- Trade-offs: There are trade-offs between the three qualities, and different methods for machine unlearning offer different trade-offs. The paper aims to compare the methods in terms of these qualities.
- Experimental study: The paper presents an experimental study of three state-of-the-art methods for machine unlearning for linear models trained with stochastic gradient descent (SGD). The methods are Fisher, Influence, and DeltaGrad, and they follow different approaches for unlearning.
- Contributions: The paper makes four contributions: (1) It defines a novel framework to compare machine unlearning methods. (2) It extends some of the existing methods with mechanisms to control performance trade-offs. (3) It offers the first experimental comparison of the methods in various settings. (4) It proposes a practical online strategy to determine when to restart the training pipeline.
- Unlearning methods: It describes three methods for machine unlearning: Fisher, DeltaGrad and Influence. Each method has a different way of injecting noise into the model parameters or the objective function to achieve certifiable unlearning.
- Evaluation metrics: It defines three metrics to measure the performance of unlearning methods: effectiveness, certifiability and efficiency. Effectiveness is the accuracy of the updated model on the test data, certifiability is the similarity of the updated model to the fully retrained model, and efficiency is the speed-up of the unlearning process compared to full retraining.
- Experimental roadmap: It provides an overview of the experiments that compare the unlearning methods on different datasets, deletion distributions, and parameter values. It also proposes a strategy to decide when to retrain the model based on the test accuracy and the estimated certifiability.
- Future directions: The authors suggest two possible directions for further research on machine unlearning. One is to extend the study to general data updates and other ML settings. The other is to develop more elaborate mechanisms to determine when a full retraining of the updated models is needed.
- References: The authors provide a list of 33 references that support their work. These include papers on machine unlearning, differential privacy, data deletion, influence functions, and superconductivity.
- Effectiveness and efficiency trade-off: The authors present the results of their experiments on different datasets and methods for different values of noise parameter 𝜎, deletion fraction 𝜀, and certifiability parameter 𝜏. They show how these parameters affect the accuracy error (AccErr) and the speed-up of the training process. They also show how their proposed methods (Fisher, DeltaGrad, and Influence) compare to the baseline methods (SGD and SGD-Noise).
- Certifiability-Effectiveness trade-off: The authors also analyze the trade-off between the certifiability and the effectiveness of their methods. They define the certifiability as the fraction of the deleted data points that are certified by the method. They show how the certifiability varies with the noise parameter 𝜎, the deletion fraction 𝜀, and the certifiability parameter 𝜏. They also show how their methods perform in terms of accuracy error (AccErr) and accuracy disparity (AccDis) for different levels of certifiability.


This is a research paper about algorithms for machine unlearning, which is the problem of deleting data points from a trained machine learning model without retraining from scratch. The paper has the following main contributions:
- It introduces a new notion of generalization in machine unlearning, where the goal is to perform well on unseen test data after data deletion, rather than minimizing the empirical loss on the remaining training data.
- It considers both computational and storage complexity of unlearning algorithms, and shows that there is a trade-off between them. It also shows that some existing unlearning algorithms based on differential privacy have suboptimal deletion capacity, which is the number of data points that can be deleted while preserving test accuracy.
- It proposes a new unlearning algorithm for convex loss functions that improves the deletion capacity by a quadratic factor in the problem dimension over differential privacy-based algorithms. It also provides theoretical guarantees on the test loss and the deletion capacity of the proposed algorithm.
The paper is related to our project, as we are also interested in developing efficient and accurate unlearning algorithms for machine learning models. The paper provides some useful insights and techniques that we can use or adapt for our own problem setting. However, the paper also has some limitations, such as:
- It only considers convex loss functions, which may not be applicable to more complex and non-convex models such as deep neural networks.
- It assumes that the unlearning algorithm has access to some cheap-to-store data statistics, which may not be available or easy to compute in some scenarios.
- It does not provide any empirical evaluation or comparison with existing unlearning algorithms on real-world datasets or applications..



To put it in simple words this algorithm focuses on making the algorithm more efficient and fast by adding noise to the sensitive data rether than deleting it. Which is similar to forgetting it. And it uses some of the mathematical functions to make that happen.
Hence the title "Remember what you wnat to forget". which is important 
