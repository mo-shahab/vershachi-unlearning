# Summary of research papers:
## Paper 1:
Author -- Yashwanth R  @https://github.com/ShiroganeMiyuki-0
## Title: NeruIPS-2021 Remember what you want to forget. 
cite: Sekhari, A., Acharya, J., Kamath, G., & Suresh, A. T. (2021). Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems, 34, 18075-18086.

link: https://proceedings.neurips.cc/paper/2021/hash/9627c45df543c816a3ddf2d8ea686a99-Abstract.html

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

## Paper 2:

## Title: A Survey Of Machine Unlearning

cite: Nguyen, T. T., Huynh, T. T., Nguyen, P. L., Liew, A. W. C., Yin, H., & Nguyen, Q. V. H. (2022). A survey of machine unlearning. arXiv preprint arXiv:2209.02299

link: https://arxiv.org/abs/2209.02299

### 1. Emphasis on Right to be Forgotten, User Privacy, and Decremental Learning:
Machine unlearning addresses the "right to be forgotten" by enabling the removal of specific data points from machine learning models, ensuring user privacy. It focuses on decremental learning, emphasizing data deletion and model repair to counter adversarial attacks.

### 2. Example of Privacy Concerns in User Behavior Data:
Consider the example of video platforms like YouTube, where vast amounts of user data, including behavior patterns, are collected. Protecting privacy in such scenarios is critical due to the sensitive nature of user interactions.

### 3. Compression of Training Data in Deep Learning Models:
Deep learning models compress intricate patterns from training data into optimized parameters. This compression ensures that essential features are retained while reducing the model's complexity, aiding accurate predictions on new data.

### 4. Adversarial Attacks in Machine Learning:
Adversarial attacks involve deliberately manipulating machine learning models to induce errors. Understanding and defending against these attacks are crucial for ensuring the integrity of machine learning systems.

### 5. Challenges and Lack of Common Frameworks:
Machine unlearning faces challenges such as the stochasticity and incrementality of training, leading to difficulties in achieving optimal results. Additionally, the lack of standardized frameworks and resources hampers the development of effective unlearning methods.

### 6. Reasons for Machine Unlearning - Security, Privacy, Usability, Fidelity:
Machine unlearning addresses security concerns, enhances user privacy, ensures system usability, and maintains fidelity in model predictions. For instance, cloud systems may unintentionally leak user data due to various factors, highlighting the need for unlearning techniques.

This below explains it with also examples

1. **Security:**<br>
Machine unlearning plays a pivotal role in ensuring the security of sensitive data. In scenarios where machine learning models are used for security-sensitive applications, such as intrusion detection or threat analysis, unlearning helps eliminate any potential vulnerabilities. By removing specific data points, especially those that might be exploited by malicious entities, machine unlearning strengthens the overall security posture of machine learning systems.

For example, in healthcare, a wrong prediction could lead to a
wrong diagnosis, a non-suitable treatment, even a death. Hence,
detecting and removing adversarial data is essential for ensuring
the model’s security and, once an attack is detected, the model
needs to be able delete the adversarial data through a machine
unlearning mechanism

2. **Privacy:**<br>
Protecting user privacy is one of the fundamental drivers behind the development of machine unlearning techniques. In applications like healthcare, finance, or personalized services, user data is often collected to enhance user experience. Machine unlearning allows for the selective removal of individual user data, ensuring that personal information remains confidential. It upholds the "right to be forgotten," enabling individuals to request the removal of their data from machine learning models, thereby safeguarding their privacy rights.

For example,
cloud systems can leak user data due to multiple copies of data hold
by different parties, backup policies, and replication strategies.

3. **Usability:**<br>
Machine unlearning contributes significantly to the usability of machine learning systems. In situations where user preferences or behaviors change over time, models need to adapt to these evolving patterns. Unlearning outdated or irrelevant data ensures that the machine learning system remains relevant and effective. This adaptability enhances user experience by ensuring that recommendations, predictions, or decisions align with current user behavior, preferences, and requirements.

For example, one can
accidentally search for an illegal product on his laptop, and find that
he keeps getting this product recommendation on this phone, even
after he cleared his web browser history.

4. **Fidelity:**<br>
Maintaining fidelity in machine learning models is crucial, especially when decisions made by these models impact individuals' lives or livelihoods. For example, in legal applications like predicting parole outcomes, machine learning models must be fair and unbiased. Unlearning helps rectify biases that might have been learned from historical data. By removing biased patterns, machine unlearning ensures that models make fair decisions, irrespective of ethnicity, gender, or other sensitive attributes. This fidelity is vital for upholding ethical standards and ensuring just and unbiased outcomes.

For example,
AI systems that have been trained on public datasets that contain
mostly white persons, such as ImageNet, are likely to make errors
when processing images of black persons. Similarly, in an application
screening system, inappropriate features, such as the gender or
race of applicants, might be unintentionally learned by the machine
learning model.

### 7. Types of Unlearning - Exact, Approximate, Zero-Glance, Zero-Shot, Few Shot:
Different unlearning scenarios exist, such as exact unlearning, which aims for perfect data removal. Approximate unlearning allows for some loss of accuracy, while zero-glance, zero-shot, and few shot unlearning involve varying degrees of forgetting with different levels of prior knowledge or examples.

`Exact unlearning`: This is when a machine learning model can forget a data item or a class exactly, without affecting the rest of the model. For example, if a model can forget a user’s data completely, without changing its performance on other users’ data, then it is an exact unlearner. Exact unlearning is often hard or impossible to achieve, as it requires a lot of computation or modification of the model.

`Approximate unlearning`: This is when a machine learning model can forget a data item or a class approximately, with some acceptable loss of accuracy or performance. For example, if a model can forget a user’s data mostly, with a small change in its performance on other users’ data, then it is an approximate unlearner. Approximate unlearning is more feasible and common than exact unlearning, as it allows for some trade-off between forgetting and accuracy.

`Zero glance unlearning`: This is when a machine learning model can forget a data item or a class without seeing any examples of that data item or class. For example, if a model can forget a user’s data by only knowing the user’s ID, then it is a zero glance unlearner. Zero glance unlearning is very challenging and rare, as it requires a lot of prior knowledge or assumptions.

`Zero shot unlearning`: This is when a machine learning model can forget a data item or a class without seeing any examples of that specific data item or class, but using some other information or data. For example, if a model can forget a user’s data by only knowing the user’s preferences or attributes, then it is a zero shot unlearner. Zero shot unlearning is less difficult than zero glance unlearning, as it uses some auxiliary information or data to guide the forgetting.

`Few shot unlearning`: This is when a machine learning model can forget a data item or a class from only a few examples of that specific data item or class. For example, if a model can forget a user’s data from only a few samples of the user’s data, then it is a few shot unlearner. Few shot unlearning is easier than zero shot unlearning, as it uses some direct information or data to guide the forgetting3

### 8. Indistinguishability Metrics and Theorems:
Indistinguishability metrics define parameters for comparing models, ensuring that unlearned models cannot leak information about the training data. These metrics are vital for evaluating the effectiveness of unlearning techniques.

### 9. Types of Unlearning Algorithms - Model-Agnostic, Model Intrinsic, Data-Driven:
Unlearning algorithms fall into categories like model-agnostic, which work across different models, and model intrinsic, tailored for specific model types such as linear, tree-based, Bayesian, or deep learning models. Data-driven methods leverage the dataset's characteristics for unlearning.

1. **Model-Agnostic Unlearning:**<br>
Model-agnostic unlearning methods are versatile and can be used with various machine learning models. They focus on removing specific data points from the training data without relying on the internal structure of the model. These methods offer a general approach that can be applied across different types of machine learning algorithms.

2. **Model Intrinsic Unlearning:**<br>
Model intrinsic unlearning algorithms are tailored to specific machine learning models. They are designed with a deep understanding of the internal workings of the model and can achieve precise unlearning outcomes. These methods are efficient and optimized for a particular model's architecture, ensuring accurate removal of data points.

3. **Data-Driven Unlearning:**<br>
Data-driven unlearning methods analyze the patterns within the training data itself. They use statistical techniques and data mining approaches to identify specific data points that need to be removed. These methods focus on understanding the inherent properties of the data, allowing for targeted and effective removal of relevant information.

### 10. Evaluation Metrics - Accuracy, Completeness, ZRF Score:
Evaluation metrics like accuracy measure the model's performance post-unlearning. Completeness assesses the thoroughness of data removal, while metrics like ZRF (Zero Recall F1) score provide a comprehensive evaluation of unlearning effectiveness.

1. ***Accuracy:***<br>
- **Definition**: Accuracy measures how well the unlearned model performs in making correct predictions after specific data points have been removed.<br>
- **Simplicity:** It gauges the overall correctness of the model's predictions post-unlearning.<br>
- **Importance:** Higher accuracy indicates that the unlearned model continues to make accurate predictions, ensuring the reliability of the modified model.

2. ***Completeness:***<br>
**Definition**: Completeness assesses the thoroughness of data removal, indicating how effectively the specified data points have been forgotten.<br>
**Simplicity:** It measures how well the unlearning process successfully removes the targeted data from the model.<br>
**Importance:** Higher completeness ensures that the unlearning process is comprehensive, leaving no trace of the specified data, thereby enhancing privacy and security.

3. ***ZRF Score (Zero Recall F1 Score):***<br>
**Definition:** ZRF score evaluates the model's ability to forget data completely. A higher ZRF score signifies better unlearning effectiveness.<br>
**Simplicity:** It quantifies how well the model forgets specific data points, with higher scores indicating stronger forgetting.<br>
**Importance:** A high ZRF score indicates that the model has successfully erased the specified data, ensuring data privacy and preventing leakage of sensitive information.

### 11. Unlearning Applications - Recommender Systems, Federated Learning, Graph Embedding, Lifelong Learning:
Machine unlearning finds applications in various domains, including recommender systems, federated learning setups, graph embedding techniques, and lifelong learning scenarios. Unlearning ensures that evolving models maintain data privacy and adaptability.

1. ***Recommender Systems:***<br>
- **Explanation:** Unlearning in recommender systems ensures that outdated or irrelevant user preferences are forgotten. This process enhances the system's ability to provide accurate and up-to-date recommendations to users.<br>
- **Importance:** Keeping recommendations relevant enhances user satisfaction and engagement, making unlearning crucial for maintaining the effectiveness of recommender systems.

2. ***Federated Learning:***<br>
- **Explanation:** In federated learning, unlearning allows models to adapt to changing data patterns across decentralized devices or servers. It ensures that outdated or biased information from specific devices is removed, improving the overall model's accuracy and fairness.<br>
- **Importance:** Unlearning in federated learning maintains the integrity of collaborative models, enhancing their adaptability and reliability across diverse data sources.

3. ***Graph Embedding:***<br>
- **Explanation:** Unlearning in graph embedding techniques involves forgetting specific relationships or nodes in a graph. This process ensures that irrelevant or outdated connections are removed, allowing the model to focus on relevant and meaningful graph structures.<br>
- **Importance:** Clean and focused graph embeddings are essential for applications like social network analysis and fraud detection, where accurate relationships are vital for decision-making.

4. **Lifelong Learning**:<br>
- **Explanation:** Lifelong learning models continually acquire new knowledge while adapting to changing data. Unlearning here allows the model to forget obsolete information, making room for new learning without interference from outdated patterns.<br>
- **Importance:** Unlearning in lifelong learning ensures that the model remains flexible and open to new information, enabling it to adapt and learn effectively over time.

### 12. Consideration of Published Datasets:
Researchers often rely on published datasets to validate unlearning techniques. These datasets provide standardized benchmarks for evaluating the performance of different unlearning algorithms.


### Note: some of the pointers i found out in the paper-

1. **About forgetting measure:<br>**
in forgetting measure there is something called a-forget, where its value 
is between 0 and 1, 0 means model remembers everything and 1 means that the model 
forgets everything

2. **Membership inference**: <br>
is a technique that tries to determine whether a data sample was used to train a model or not, based on the model’s output. If a model forgets a data sample completely, then membership inference should not be able to distinguish it from a new sample. Therefore, the forgetting rate is defined as the percentage of data samples that become indistinguishable from unknown samples after unlearning. The higher the forgetting rate, the better the unlearning method. For example, if a model has a forgetting rate of 90%, it means that 90% of the unlearned data samples are no longer memorized by the model, and the model’s privacy is improved

3. **what does it mean by "the model leaks data" ?**:<br>
The model leaks data means that the model reveals some information about the data that it was trained on, which could compromise the privacy and security of the data owners. For example, if the model is trained on sensitive medical records, and someone can infer whether a person was in the training data or not by querying the model, then the model leaks data. Data leakage can also affect the performance and generalization of the model, as it may overfit to the training data and fail to adapt to new data. Data leakage can happen due to various reasons, such as improper data splitting, feature engineering, or preprocessing. To prevent data leakage, one should ensure that the training and testing data are independent and representative of the real-world data, and that the features used for the model do not contain information that is unavailable at prediction time

4. **back door attacks:** <br> deceiving the machine learning models

### About the pre-trained models for our tool, have a look at this: 
| Unlearning Algos | Language | Platform | Applicable ml models | code repo |
| Unlearning Algorithm | Language   | Platform               | Applicable ML Models      | Code Repository                                              |
|----------------------|------------|------------------------|---------------------------|--------------------------------------------------------------|
| SISA [11]            | Python     | -                      | Model-agnostic            | https://github.com/cleverhans-lab/machine-unlearning |
| Athena [142, 143]    | Python     | -                      | Model-agnostic            | https://github.com/inspire-group/unlearning-verification |
| AmnesiacML [58]      | Python     | -                      | Model-agnostic            | https://github.com/lmgraves/AmnesiacML               |
| Kpriors [80]         | Python     | PyTorch                | Model-agnostic            | https://github.com/team-approx-bayes/kpriors         |
| ERM [109]            | Python     | -                      | Model-agnostic            | https://github.com/ChrisWaites/descent-to-delete      |
| ShallowAttack [23]   | Python     | PyTorch                | Model-agnostic            | https://github.com/MinChen00/UnlearningLeaks         |
| UnrollingSGD [148]   | Python     | -                      | Model-agnostic            | https://github.com/cleverhans-lab/unrolling-sgd      |
| DeltaGrad [170]      | Python     | -                      | Model-agnostic            | https://github.com/thuwuyinjun/DeltaGrad             |
| Amnesia [131]        | Rust       | -                      | Model-agnostic            | https://github.com/schelterlabs/projects-amnesia     |
| MUPy [13]            | Python     | LensKit kNN            | kNN                       | https://github.com/theLauA/MachineUnlearningPy       |
| DelKMeans [52]       | Python     | -                      | kMeans                    | https://github.com/tginart/deletion-efficient-kmeans |
| CertifiedRem [59]    | Python     | PyTorch                | Linear models             | https://github.com/facebookresearch/certified-removal |
| CertAttack [100]     | Python     | Tensorflow             | Linear models             | https://github.com/ngmarchant/attack-unlearning      |
| PRU [73]             | Python     | -                      | Linear models             | https://github.com/zleizzo/datadeletion               |
| HedgeCut [132]       | Python     | -                      | Tree-based models         | https://github.com/schelterlabs/hedgecut              |
| DaRE-RF [12]         | Python     | -                      | Tree-based models         | https://github.com/jjbrophy47/dare_rf                |
| MCMC-Unlearning [45] | Python     | PyTorch                | Bayesian models           | https://github.com/fshp971/mcmc-unlearning            |
| BIF [46]             | Python     | PyTorch                | Bayesian models           | https://github.com/fshp971/BIF                        |
| L-CODEC [105]        | Python, Matlab | PyTorch              | Deep learning             | https://github.com/vsingh-group/LCODEC-deep-unlearning |
| SelectiveForgetting [55, 56] | Python | -                   | Deep learning             | https://github.com/AdityaGolatkar/SelectiveForgetting |
| Neurons [30]         | Python     | -                      | Deep learning             | https://github.com/Hunter-DDM/knowledge-neurons       |
| Unlearnable [70]     | Python     | -                      | Deep learning             | https://github.com/HanxunH/Unlearnable-Examples       |
| DLMA [173]           | Python     | -                      | Deep learning             | https://github.com/AnonymousDLMA/MI_with_DA           |
| GraphProjector [29]  | Python     | -                      | Graph Learning            | https://anonymous.4open.science/r/Projector-NeurIPS22/README.md |
| GraphEditor [28]     | Python     | -                      | Graph Learning            | https://anonymous.4open.science/r/GraphEditor-NeurIPS22-856E/README.md |
| RecEraser [19]       | Python, C++ | -                    | Recommender Systems       | https://github.com/chenchongthu/Recommendation-Unlearning |
| FedEraser [90]       | Python     | -                      | Federated Learning         | https://www.dropbox.com/s/1lhx962axovbbom/FedEraser-Code.zip?dl=0 |
| RapidFed [95]        | Python     | -                      | Federated Learning         | https://github.com/yiliucs/federated-unlearning       |
