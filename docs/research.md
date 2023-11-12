# Summary of research papers:
## Paper 1: (algorithms of mul)
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

## Paper 2: (applications, algorithms, perfomance of mul)

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

### 7. Types of requests that an user can make to an mul framework:
- Item Removal:<br>
Explanation: Users can request the removal of specific data points or items from the trained model. This means the model forgets these particular pieces of information.<br>
Example: If a user's data point is removed, the model will no longer consider it during predictions.

- Feature Removal:<br>
Explanation: Users can ask to forget certain features or attributes used for training. Removing features ensures the model doesn't rely on specific input characteristics.<br>
Example: If a feature representing age is removed, the model won't consider age information for predictions anymore.

- Class Removal:<br>
Explanation: Users can request the removal of entire classes or categories from the model. This action makes the model forget how to classify certain types of data.<br>
Example: If a class representing 'dogs' is removed, the model will no longer recognize or classify any input as 'dogs'.

- Task Removal:<br>
Explanation: Users can ask to forget a specific task or prediction objective the model was trained for. This allows the model to focus on other tasks.<br>
Example: If a model was trained for both image recognition and text analysis, removing the 'text analysis' task means the model will only perform image recognition.

- Stream Removal:<br>
Explanation: Users can request the removal of a continuous data stream or source. This unlearning action ensures the model doesn't learn from ongoing data input.<br>
Example: If a data stream from a particular sensor is removed, the model will no longer update its knowledge based on new data from that sensor.

### 8. Types of Unlearning - Exact, Approximate, Zero-Glance, Zero-Shot, Few Shot:
Different unlearning scenarios exist, such as exact unlearning, which aims for perfect data removal. Approximate unlearning allows for some loss of accuracy, while zero-glance, zero-shot, and few shot unlearning involve varying degrees of forgetting with different levels of prior knowledge or examples.

`Exact unlearning`: This is when a machine learning model can forget a data item or a class exactly, without affecting the rest of the model. For example, if a model can forget a user’s data completely, without changing its performance on other users’ data, then it is an exact unlearner. Exact unlearning is often hard or impossible to achieve, as it requires a lot of computation or modification of the model.

`Approximate unlearning`: This is when a machine learning model can forget a data item or a class approximately, with some acceptable loss of accuracy or performance. For example, if a model can forget a user’s data mostly, with a small change in its performance on other users’ data, then it is an approximate unlearner. Approximate unlearning is more feasible and common than exact unlearning, as it allows for some trade-off between forgetting and accuracy.

`Zero glance unlearning`: This is when a machine learning model can forget a data item or a class without seeing any examples of that data item or class. For example, if a model can forget a user’s data by only knowing the user’s ID, then it is a zero glance unlearner. Zero glance unlearning is very challenging and rare, as it requires a lot of prior knowledge or assumptions.

`Zero shot unlearning`: This is when a machine learning model can forget a data item or a class without seeing any examples of that specific data item or class, but using some other information or data. For example, if a model can forget a user’s data by only knowing the user’s preferences or attributes, then it is a zero shot unlearner. Zero shot unlearning is less difficult than zero glance unlearning, as it uses some auxiliary information or data to guide the forgetting.

`Few shot unlearning`: This is when a machine learning model can forget a data item or a class from only a few examples of that specific data item or class. For example, if a model can forget a user’s data from only a few samples of the user’s data, then it is a few shot unlearner. Few shot unlearning is easier than zero shot unlearning, as it uses some direct information or data to guide the forgetting3

### 9. Indistinguishability Metrics and Theorems:
Indistinguishability metrics define parameters for comparing models, ensuring that unlearned models cannot leak information about the training data. These metrics are vital for evaluating the effectiveness of unlearning techniques.

### 10. Types of Unlearning Algorithms - Model-Agnostic, Model Intrinsic, Data-Driven:
Unlearning algorithms fall into categories like model-agnostic, which work across different models, and model intrinsic, tailored for specific model types such as linear, tree-based, Bayesian, or deep learning models. Data-driven methods leverage the dataset's characteristics for unlearning.

1. **Model-Agnostic Unlearning:**<br>
Model-agnostic unlearning methods are versatile and can be used with various machine learning models. They focus on removing specific data points from the training data without relying on the internal structure of the model. These methods offer a general approach that can be applied across different types of machine learning algorithms.

2. **Model Intrinsic Unlearning:**<br>
Model intrinsic unlearning algorithms are tailored to specific machine learning models. They are designed with a deep understanding of the internal workings of the model and can achieve precise unlearning outcomes. These methods are efficient and optimized for a particular model's architecture, ensuring accurate removal of data points.

3. **Data-Driven Unlearning:**<br>
Data-driven unlearning methods analyze the patterns within the training data itself. They use statistical techniques and data mining approaches to identify specific data points that need to be removed. These methods focus on understanding the inherent properties of the data, allowing for targeted and effective removal of relevant information.

### 11. Different verifications on MUL:
- Feature Injection Test:<br>
Explanation: This test involves injecting specific features into the model to observe if it has truly forgotten those features. If the model reacts to these features, it suggests incomplete unlearning.<br>
Example: Injecting a removed feature (like a deleted user attribute) and checking if the model responds to it.

- Forgetting Measure:<br>
Explanation: Forgetting measure quantifies the extent to which the model has forgotten specific data points. A higher forgetting rate indicates better unlearning.<br>
Example: If the model has a forgetting rate of 90%, it means 90% of the unlearned data samples are no longer memorized.

- Membership Inference Attacks:<br>
Explanation: Attackers attempt to determine if a particular data sample was used for training. Effective unlearning should make it difficult to distinguish between used and unused data.<br>
Example: Trying to identify if a specific image was part of the training dataset by querying the model's responses.

- Slow Down Attacks:<br>
Explanation: Slow down attacks measure the model's response time. Unlearning should not significantly slow down the model’s prediction process.<br>
Example: Monitoring the time taken by the model to respond to queries before and after unlearning.

- Interclass Confusion Test:<br>
Explanation: This test assesses if the model confuses removed classes with existing ones. Proper unlearning ensures that deleted classes don’t affect the recognition of remaining classes.<br>
Example: Checking if the model mistakenly identifies a removed category as a similar existing category.

- Federated Verification:<br>
Explanation: Federated verification assesses unlearning in a distributed environment where models are trained across multiple devices. It ensures that unlearning works seamlessly in federated learning setups.<br>
Example: Verifying if models in a federated system correctly unlearn data across all devices involved in training.

- Cryptographic Protocol:<br>
Explanation: Cryptographic methods can be used to validate unlearning processes securely. These protocols ensure that unlearning actions are authenticated and authorized.<br>
Example: Implementing encryption techniques to secure communication between the unlearning system components.

### 11. Evaluation Metrics - Accuracy, Completeness, ZRF Score:
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

### 12. Unlearning Applications - Recommender Systems, Federated Learning, Graph Embedding, Lifelong Learning:
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

### 13. Consideration of Published Datasets:
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

### References made in this paper that we can make use of, with citation:

These are the papers that are referenced in that particular section of the paper.
1. Introduction:
- Yinzhi Cao and Junfeng Yang. 2015. Towards making systems forget with
machine unlearning. In 2015 IEEE Symposium on Security and Privacy. 463–480.
  - Things mentioned in the above paper are:
    - Reasons for machine unlearning
    - The need for the systems to forget

- Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram
Galstyan. 2021. A survey on bias and fairness in machine learning. CSUR 54, 6
(2021), 1–35.
  - Things mentioned in this paper
    - the bias in machine learning, for example in a beauty contest they used the ai as the judge and ai chose fair peopleover someone that had darker skin tone, and also how there was a bias against the asian people.

2. Challenges in the machine unlearning
- Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui
Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. 2021.
Machine unlearning. In SP. 141–159.
  - Things mentioned in this paper ( important paper ): 
    - The challenges in the machine unlearning
    - keywords like "stochasticity of training" and "incrementality of training" are mentioned

3. Unlearning requests:
- Alexander Warnecke, Lukas Pirch, Christian Wressnegger, and Konrad
Rieck. 2021. Machine Unlearning of Features and Labels. arXiv preprint
arXiv:2108.11577 (2021).
  - Things mentioned:
    - different types of the requests that can be made in machine unlearning

4. Design requirements in the MUL:
- Yinzhi Cao and Junfeng Yang. 2015. Towards making systems forget with
machine unlearning. In 2015 IEEE Symposium on Security and Privacy. 463–480.
  - Things mentioned:
    - different types of requirements such as completeness, accuracy, timeliness and such is mentioned
    - To prepare for unlearning process, many techniques
need to store model checkpoints, historical model updates, training
data, and other temporary data
    - The requirement of the framework to be model agnostic is mentioned

- Yingzhe He, Guozhu Meng, Kai Chen, Jinwen He, and Xingbo Hu. 2021. Deepobliviate:
a powerful charm for erasing data residual memory in deep neural
networks. arXiv preprint arXiv:2105.06209 (2021).
  - Things mentioned:
    - the accuracy of the unlearned model is often
measured on a new test set, or it is compared with that of the
original model before unlearning
    - To prepare for unlearning process, many techniques
need to store model checkpoints, historical model updates, training
data, and other temporary data

5. Verification in unlearnign:
- Anvith Thudi, Hengrui Jia, Ilia Shumailov, and Nicolas Papernot. 2022. On
the necessity of auditable algorithmic definitions for machine unlearning. In
USENIX Security. 4007–4022.
  - things mentioned:
    - The goal of unlearning verification methods is to certify that one
cannot easily distinguish between the unlearned models and their
retrained counterparts

6. Types of machine unlearning:
- Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui
Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. 2021.
Machine unlearning. In SP. 141–159.
- Jonathan Brophy and Daniel Lowd. 2021. Machine unlearning for random
forests. In ICML. 1092–1104.
- Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. 2022. Unrolling sgd: Understanding factors influencing machine unlearning. In EuroS&P. 303–319.
  - things mentioned:
    - different types of machine unlearning, exact, approximate, and such.

7. Unlearning scenarios:
- Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. 2022. Unrolling sgd: Understanding factors influencing machine unlearning. In EuroS&P. 303–319.
  - things mentioned:
    - zero glance unlearning, zero shot unlearning
    
- Youngsik Yoon, Jinhwan Nam, Hyojeong Yun, Dongwoo Kim, and Jungseul Ok. 2022. Few-Shot Unlearning by Model Inversion. arXiv preprint arXiv:2205.15567 (2022).
  - things mentioned:
    - few shot unlearning


8. Unlearning algorithms:
the different types of unlearning algorithms, with their papers:
- SISA
  - Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. 2021. Machine unlearning. In SP. 141–159.

- Athena
  - David Marco Sommer, Liwei Song, Sameer Wagh, and Prateek Mittal. 2020.
Towards probabilistic verification of machine unlearning. arXiv preprint
arXiv:2003.04247 (2020).
  - David Marco Sommer, Liwei Song, Sameer Wagh, and Prateek Mittal. 2022.
Athena: Probabilistic Verification of Machine Unlearning. Proc. Priv. Enhancing
Technol. 2022, 3 (2022), 268–290.

- AmnesiacML
  - Laura Graves, Vineel Nagisetty, and Vijay Ganesh. 2021. Amnesiac machine
learning. In AAAI, Vol. 35. 11516–11524.

- Kpriors
  - Mohammad Emtiyaz E Khan and Siddharth Swaroop. 2021. Knowledgeadaptation
priors. NIPS 34 (2021), 19757–19770.

- ERM
  - Seth Neel, Aaron Roth, and Saeed Sharifi-Malvajerdi. 2021. Descent-to-delete:
Gradient-based methods for machine unlearning. In Algorithmic Learning Theory.
931–962.

- ShallowAttack
  - Min Chen, Zhikun Zhang, Tianhao Wang, Michael Backes, Mathias Humbert, and Yang Zhang. 2021. When machine unlearning jeopardizes privacy. In SIGSAC. 896–911.

- UnrollingSGD
  - Anvith Thudi, Gabriel Deza, Varun Chandrasekaran, and Nicolas Papernot. 2022. Unrolling sgd: Understanding factors influencing machine unlearning. In EuroS&P. 303–319.

- DeltaGrad
  - Yinjun Wu, Edgar Dobriban, and Susan Davidson. 2020. Deltagrad: Rapid
retraining of machine learning models. In ICML. 10355–10366.

- Amnesia
  - Sebastian Schelter. 2020. “Amnesia” - A Selection of Machine Learning Models That Can Forget User Data Very Fast. In CIDR.

- MUPy
  - Yinzhi Cao and Junfeng Yang. 2015. Towards making systems forget with machine unlearning. In 2015 IEEE Symposium on Security and Privacy. 463–480.

- DelKMeans
  - Antonio Ginart, Melody Guan, Gregory Valiant, and James Y Zou. 2019. Making ai forget you: Data deletion in machine learning. NIPS 32 (2019).

- CertifiedRem
  - Chuan Guo, Tom Goldstein, Awni Y. Hannun, and Laurens van der Maaten. 2020. Certified Data Removal from Machine Learning Models. In ICML, Vol. 119. 3832–3842.

- CertAttack
  - Neil G Marchant, Benjamin IP Rubinstein, and Scott Alfeld. 2022. Hard to Forget: Poisoning Attacks on Certified Machine Unlearning. In AAAI, Vol. 36. 7691–7700.

- PRU
  - Zachary Izzo, Mary Anne Smart, Kamalika Chaudhuri, and James Zou. 2021. Approximate data deletion from machine learning models. In AISTAT. 2008–2016.

- HedgeCut
  - Sebastian Schelter, Stefan Grafberger, and Ted Dunning. 2021. Hedgecut: Maintaining randomised trees for low-latency machine unlearning. In SIGMOD. 1545–1557.

- DaRE-RF
  - Jonathan Brophy and Daniel Lowd. 2021. Machine unlearning for random forests. In ICML. 1092–1104.

- MCMC-Unlearning
  - Shaopeng Fu, Fengxiang He, and Dacheng Tao. 2022. Knowledge Removal in Sampling-based Bayesian Inference. In ICLR.

- BIF
  - Shaopeng Fu, Fengxiang He, Yue Xu, and Dacheng Tao. 2021. Bayesian inference forgetting. arXiv preprint arXiv:2101.06417 (2021).

- L-CODEC
  - Ronak Mehta, Sourav Pal, Vikas Singh, and Sathya N Ravi. 2022. Deep Unlearning via Randomized Conditionally Independent Hessians. In CVPR. 10422–10431.

- SelectiveForgetting
  - Aditya Golatkar, Alessandro Achille, and Stefano Soatto. 2020. Eternal sunshine of the spotless net: Selective forgetting in deep networks. In CVPR. 9304–9312.
  - Aditya Golatkar, Alessandro Achille, and Stefano Soatto. 2020. Forgetting outside the box: Scrubbing deep networks of information accessible from input-output

- Neurons
  - Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge Neurons in Pretrained Transformers. In ACL. 8493–8502.

- Unlearnable
  - Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, and Yisen Wang. 2021. Unlearnable Examples: Making Personal Data Unexploitable. In ICLR.

- DLMA
  - Da Yu, Huishuai Zhang, Wei Chen, Jian Yin, and Tie-Yan Liu. 2021. How does data augmentation affect privacy in machine learning?. In AAAI, Vol. 35. 10746–10753.

- GraphProjector
  - Weilin Cong and Mehrdad Mahdavi. [n. d.]. Privacy Matters! Efficient Graph Representation Unlearning with Data Removal Guarantee. ([n. d.]).

- GraphEditor
  - Weilin Cong and Mehrdad Mahdavi. [n. d.]. GRAPHEDITOR: An Efficient Graph Representation Learning and Unlearning Approach. ([n. d.]).

- RecEraser
  - Chong Chen, Fei Sun, Min Zhang, and Bolin Ding. 2022. Recommendation unlearning. In WWW. 2768–2777.

- FedEraser
  - Gaoyang Liu, Xiaoqiang Ma, Yang Yang, Chen Wang, and Jiangchuan Liu. 2021. Federaser: Enabling efficient client-level data removal from federated learning models. In IWQOS. 1–10.

- RapidFed
  - Yi Liu, Lei Xu, Xingliang Yuan, Cong Wang, and Bo Li. 2022. The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining. In INFOCOM. 1749–1758.

## Paper 3: (algorithms of mul)
Author -- Yashwanth R  @https://github.com/ShiroganeMiyuki-0
## Title: Coded Machine Unlearning

cite: Aldaghri, N., Mahdavifar, H., & Beirami, A. (2021). Coded machine unlearning. IEEE Access, 9, 88137-88150.

link: https://ieeexplore.ieee.org/document/9458237

- Coded machine unlearning: A new framework for removing data from trained ML models using data encoding and ensemble learning. It aims to achieve perfect unlearning with better performance and lower cost than uncoded methods.
- Problem setup: A regression problem with a training dataset of n samples and d features. The loss function is regularized and kernelized. The model uses random projections to reduce the dimensionality of the features. The unlearning protocol must satisfy the perfect unlearning criterion.
- Learning protocol: The model uses a linear encoder to transform the training dataset into r coded shards, each with n/s samples. The shards are assigned to r weak learners that are trained independently and aggregated using an averaging function. The encoder usa random binary matrix with rate τ = n/m and density ρ.
- Unlearning protocol: The model identifies the coded shards that contain the sample to be unlearned and updates them by subtracting the sample from the corresponding coded samples. Then, the model retrains the affected weak learners and updates the aggregate model. The protocol guarantees perfect unlearning.
- Synthetic data experiments: The authors test their coded learning and unlearning protocol on three synthetic datasets with different features and response variables. They show that coding provides a better trade-off between performance and unlearning cost for datasets with heavy-tailed features, but not for datasets with normal features.
- Conclusion: The authors summarize their main contributions and discuss some possible directions for future work, such as extending the protocol to other models, studying different classes of codes, and exploring the role of influential samples in coded learning and unlearning.

- The algorithm in this paper is about coded machine unlearning, which is a method to efficiently remove the information of a sample from a trained machine learning model. The algorithm consists of two parts: learning and unlearning.
- Learning: The algorithm uses an encoder to transform the original training dataset into a smaller number of coded shards, which are then used to train weak learners independently. The final model is obtained by aggregating the weak learners' models using a certain function, such as averaging.
- Unlearning: The algorithm identifies the coded shards that contain the sample to be unlearned and updates them by subtracting the sample from the corresponding coded samples. Then, the algorithm retrains the affected weak learners using the updated coded shards and updates the final model by recalculating the aggregation function. The algorithm guarantees perfect unlearning, which means that the updated model is equivalent to a model trained on the dataset without the unlearned sample..
 
### How this model is different than the first?
- The overall difference between the two parts of the page is that the first part focuses on the problem of coded machine unlearning, which is a new framework for removing data from trained ML models using data encoding and ensemble learning, while the second part focuses on the problem of probabilistic machine unlearning, which is a relaxed definition of data deletion that requires the output model to be similar to the one trained without the deleted data. The first part proposes a coded learning and unlearning protocol that uses random linear coding to combine the training samples into smaller coded shards and updates the model accordingly. The second part proposes a Gaussian mechanism for unlearning that adds Gaussian noise to the output of the learning algorithm and proves its differential privacy and excess risk guarantees. The first part also presents some synthetic data experiments to demonstrate the performance versus unlearning cost trade-off of the coded protocol. The second part also derives a lower bound on the sample size required for unlearning and extends the results to the case of convex but not strongly convex loss functions.

## Paper 4: (applications of mul)

## Title: Toward Highly-Efficient and Accurate Services QoS Prediction via Machine Unlearning

cite: Zeng, Y., Xu, J., Li, Y., Chen, C., Dai, Q., & Du, Z. (2023). Towards Highly-efficient and Accurate Services QoS Prediction via Machine Unlearning. IEEE Access.

link: https://ieeexplore.ieee.org/abstract/document/10171348

### Note: this paper has no direct relevance with our framework.

- **CADDEraser Framework:**<br>
  - The paper introduces CADDEraser, a new system designed to handle requests from users who want their data removed from IoT services. CADDEraser is efficient and ensures accurate predictions while addressing the challenges related to data removal (unlearning requests).

- **Handling User Data Sensibly:**<br>
  - IoT devices need to handle user data responsibly. If not handled carefully, it can lead to issues like data contamination, where incorrect or irrelevant data affects service predictions. CADDEraser helps in managing this challenge effectively.

- **Use of Machine Unlearning (MUL)**:<br>
  - Machine Unlearning (MUL) is a technique used here. MUL helps in removing personal data from the system. In the context of this paper, MUL is used to improve the quality of predictions made by IoT services.

- **Importance of QoS Prediction:**<br>
  - Quality of Service (QoS) prediction is crucial for service providers. It affects market share and user retention. Deep learning models, although powerful, can be vulnerable to attacks that compromise user data. Machine Unlearning is applied here to remove sensitive data and maintain the accuracy of predictions.

- **Preventing Performance Degradation:**<br>
  - When risky or irrelevant data is present, it can degrade the performance of prediction models. To prevent this, CADDEraser erases such data and retrains the model using a cleaned dataset. This ensures that the predictions remain accurate and reliable.

- **Relevance to Our Research:**<br>
  - This paper aligns with our research focus on machine unlearning. While their primary concern is QoS prediction, they utilize machine unlearning techniques to enhance their predictions. Similarly, in our research, we're working on machine unlearning tools. The methods and challenges discussed in this paper provide valuable insights for our own work in the field of machine unlearning.
In summary, this paper's approach is relevant to our research because it showcases the practical use of machine unlearning techniques in improving the accuracy and reliability of predictions in IoT services. The challenges they address and the methods they employ provide useful context and inspiration for our own machine unlearning project.

## Paper 5: (algorithms of mul)

## Title: Approximate Data Deletion from Machine Learning Models

cite: Izzo, Z., Smart, M. A., Chaudhuri, K., & Zou, J. (2021, March). Approximate data deletion from machine learning models. In International Conference on Artificial Intelligence and Statistics (pp. 2008-2016). PMLR.

link: https://proceedings.mlr.press/v130/izzo21a.html

- In this paper, the focus revolves around crucial data privacy regulations such as the EU's General Data Protection Regulation and the California Consumer Privacy Act. These regulations empower individuals to request the removal of their personal data from systems operated by large companies like Google and Facebook, emphasizing the fundamental concept of the "right to be forgotten."

- Recent research has revealed the vulnerability of machine learning models, especially in vision and natural language processing (NLP) domains, where attackers can reconstruct training data. The paper addresses a critical challenge: how to efficiently and accurately delete specific batches of data points (referred to as \(k\) points) from a precomputed machine learning model. This challenge arises because once a model is prepared, removing data points without compromising its efficiency becomes a complex task.

- Exact data deletion, while desirable, is computationally intensive, making it impractical for real-world applications. To tackle this challenge, the paper delves into the realm of approximate unlearning, exploring innovative methods to efficiently handle deletion requests without compromising the model's overall accuracy.

- One notable technique introduced in the paper is Newton's method, which offers an approximate approach to data retraining. By leveraging this method, the researchers aim to strike a balance between computational efficiency and accuracy when updating the model after data deletion.

- Additionally, the paper discusses influential methods within the context of model retraining, comparing and contrasting them with the exact deletion approach. These discussions provide valuable insights into the trade-offs between precision and computational complexity when dealing with data removal requests.

In summary, the paper outlines the challenges posed by data deletion requests within the framework of stringent privacy regulations. It proposes an innovative computational model inspired by these challenges, emphasizing the need for efficient and accurate data removal techniques. The incorporation of Newton's method and in-depth discussions about influential methods enrich the exploration of approximate unlearning, paving the way for more effective solutions in the realm of machine learning privacy.

### the difference between exact, newton's, influential method and approximate method with examples

 - Imagine you have a smart system, like those used by big companies such as Google and Facebook. These systems store lots of personal data. People have the right to request their data to be removed, ensuring their privacy. This is a fundamental concept known as the "right to be forgotten," protected by laws like the EU's General Data Protection Regulation and the California Consumer Privacy Act.

- Now, the challenge arises when you want to remove specific pieces of data from this smart system. Think of these pieces of data as 'k points.' Deleting them while keeping the system working efficiently is tough once it's set up.

- One way to do this is the 'exact method.' It's like surgically removing the specific data points without harming the rest of the system. But, doing it precisely requires a lot of computational power, making it slow and impractical for real-world situations.

- To tackle this, researchers explore 'approximate unlearning.' It's like finding a balance between speed and accuracy. One approach they introduce is 'Newton's method.' Imagine you have a painting, and you want to modify it slightly without ruining the whole artwork. Newton's method helps in making those small, precise changes efficiently.

- Additionally, the paper discusses 'influential methods.' These are techniques that guide how the system learns and evolves. They are like experienced teachers helping a student understand complex topics. By comparing these methods with the exact deletion approach, the researchers explore which methods work best for efficient and accurate data removal.

## Paper 6: (overview of unlearning)

## Title: Fast Yet Effective Machine Unlearning

Cite: Tarun, A. K., Chundawat, V. S., Mandal, M., & Kankanhalli, M. (2023). Fast yet effective machine unlearning. IEEE Transactions on Neural Networks and Learning Systems. 
arXiv:2111.08947 [cs.LG]

1.	Introduction
   
Unlearning the data observed during the training of a machine learning (ML) model is an important task that can play a pivotal role in fortifying the privacy and security of ML-based applications.
Privacy regulations are increasing day by day to include provisions in future to give the control of personal privacy to the individuals. That is for instance the California Consumer Privacy Act (CCPA) allows companies to collect the user data by default. However, the user has the right to delete it on request.
This also implies that the company is requested to remove the data for example face id of a user (or a set of users) from the already trained face recognition model. In addition, there is a constraint such that the company no longer has access to those (requested to be removed) facial images.
The unlearning that is selective forgetting or data deletion solutions presented in the literature are focused on simple learning algorithms such a linear/logistic regression, random forests, and k-means clustering and other analysis. 
Initial work on forgetting in convolutional networks is however shown to be effective only on small scale problems and are computationally expensive due to non-convex loss functions.

2.	Open problems on efficient unlearning in deep networks are
   
•	efficiently unlearning multiple classes is yet to be explored due to several complexities that arise while working with deep learning models that is optimization and final network weight combination.
•	several optimal set of weights may exist for the same network, making it difficult to confidently evaluate the degree of unlearning.
•	Forgetting a large part of data or an entire class of data while preserving the accuracy of the model is a major problem
•	Moreover, efficiently manipulating the network weights without using the unlearning data still remains an unsolved problem. 
In brief the main problems that arise are to unlearn multiple classes of data, perform unlearning for large-scale problems, and generalize the solution to different type of deep network.

3.	Proposed by Authors
   
In this paper, they propose a framework for unlearning in a zero-glance privacy setting, i.e. the model can’t see the unlearning class of data we learn an error-maximizing noise matrix consisting of highly influential points corresponding to the unlearning class. Then, they train the model using the noise matrix to update the network weights and introduce Unlearning by Selective Impair and Repair (UNSIR), a single-pass method to unlearn single/multiple classes of data in a deep model without requiring access to the data samples of the requested set of unlearning classes.

4.	Comparison Models
Models used for comparison are ResNet18, AllCNN, MobileNetv2 and Vision Transformers

5.	Used Datasets
Datasets used CIFAR-10 [53], CIFAR-100 [53] and VGGFace-100.

6.	Might require later for analysis
   
(Let the complete training dataset consisting of n samples and K total number of classes be Dc = f(xi;yi)gni=1where x 2 X   Rd are the inputs and y 2 Y = 1; :::;K are the corresponding class labels. If the forget and retain classes are denoted by Yf and Yr then Df [Dr = Dc, Df \Dr = ;. Let the deep learning model be represented by the function f (x) : X ! Y parameterized by   2 Rd used to model the relation X ! Y. The weights   of the original trained deep network f  1 are a function of the complete training data Dc. Forgetting in zero-glance privacy setting is an algorithm, which gives a new set of weights  Dr sub by using the trained model f and a subset of retain images Dr sub   Dr which doesn’t remember the information regarding Df and behaves similarly to a model which has never seen Df in the parameter and output space. To achieve unlearning, we first learn a noise matrix N for each class in Yf by using the trained model. Then we transform the model in such a way that it fails to classify the samples from forget set Df while maintaining the accuracy for classifying the samples from the retain set Dr. This is ensured by using a small subset of samples Dr sub drawn from the retain dataset Dr)

7.	ERROR-MAXIMIZING NOISE BASED UNLEARNING
   
The approach aims to learn a noise matrix for the unlearning class by maximizing the model loss. Such generated noise samples will damage/overwrite the previously learned network weights for the relevant classes during the model update and induce unlearning. Error maximizing noise will have high influence to enable parameters updates corresponding to the unlearning class.

A. Error-maximizing Noise
The goal is to create a correlation between Noise N and the unlearning class label f.

B. UNSIR: Unlearning with Single Pass Impair and Repair
  •	Impair. 
    They train the model on a small subset of data from the original distribution which also contains generated noise. This     step is called ’impair’ as it corrupts those weights in the network which are responsible for recognition of the data       in forget class. They use a high learning rate and observe that almost always only a single epoch of ’impair’ is enough.
  •	Repair. 
    The ’impair’ step may sometimes disturb the weights that are responsible for predicting the retain classes. Thus, they      ’repair’ those weights by training the model for a single epoch (on rare occasions, more epochs may be required) on the     retain data (Dr sub).

8.	Evaluation Metrics
    
•	Accuracy on forget set (ADf ): Should be close to zero.
•	Accuracy on retain set (ADr ): Should be close to the performance of original model.
•	Relearn time (RT): Relearn time is a good proxy to measure the amount of information remaining in the model about the unlearning data. 
•	Weight distance: The distance between individual layers of the original model and the unlearned model gives additional insights about the amount of information remaining in the network about the forget data.
•	Prediction distribution on forget class: We analyze the distribution of the predictions for different samples in the forget classes of data in the unlearned model. Presence of any specific observable patterns like repeatedly predicting a single retain class may indicate risk of information exposure.

9.	ANALYSIS

A.	Layer-wise Distance between the Network Weights
B. Efficiency
C. Sequential Forgetting
D. Visualizing the Unlearning in Models
E. Effect of Different Learning Rates in UNSIR
F. On the Validity of Retrained Model as the Gold Standard
G. Using Different Proportions of Retain Data (Dr) for Unlearning
H. Different Levels of Weight Penalization
I. Healing after Multiple Steps of Repair

10.	Results
    
The results were compared with three baseline unlearning methods: Retrain Model, FineTune, and NegGrad. Due to poor results of FineTune, NegGrad, and Fisher forgetting in CIFAR-10, we compare our results only with the Retrain Model in the subsequent experiments.
•	Single Class Unlearning:
The proposed model is able to erase the information with respect to a particular class and unlearn in a single shot of impair and repair. They were able to obtain superior accuracy in retain set (Dr) and forget set (Df ) over the existing methods. The relearn time (RT) is much higher for our method in comparison to the baseline methods.

•	Multiple Class Unlearning: 
Our method shows excellent results for unlearning multiple classes of data. We observe that with the increase in the number of classes to unlearn, the repair step becomes more effective and leads to performance closer to the original model on (Dr) set.

## Paper 7: (application of mul)
## Title: Lifelong Anomaly Detection Through Unlearning

cite: Min Du, Zhi Chen, Chang Liu, Rajvardhan Oak, and Dawn Song. 2019. Lifelong Anomaly Detection Through Unlearning. In 2019 ACM SIGSAC Conference on Computer and Communications Security (CCS ’19), November 11–15, 2019, London, United Kingdom. ACM, New York, NY, USA, 15 pages.

link: https://dl.acm.org/doi/abs/10.1145/3319535.3363226

### What is Anamoly Detection?
- so basically anamoly detection, also known as outlier detection, it is just a technique used in data mining and ml, that do not conform expected behaviour within a dataset. the unusual patterns are referred to as the anamolies.

### What is Lifelong Anamoly Detection?
- Lifelong anomaly detection refers to the process of continuously learning and adapting to new patterns of anomalies over time. Unlike traditional anomaly detection methods that assume a static environment, lifelong anomaly detection systems are designed to handle evolving data distributions and changing patterns of anomalies that occur in dynamic and continuously changing datasets.

- In lifelong anomaly detection, the model is capable of learning from new data and adapting to new types of anomalies as they emerge. This approach is particularly useful in applications where the characteristics of anomalies can change over time or where the system needs to continuously monitor and detect novel threats or irregularities.

- The key idea behind lifelong anomaly detection is to create adaptive models that can evolve and improve their performance over the long term, ensuring accurate anomaly detection even in the face of changing data patterns and emerging anomalies. This approach enhances the system's resilience and effectiveness in real-world scenarios where the environment is constantly evolving.

### How is this paper related to the unlearning ? 

- Imagine you have a smart system that's continuously learning from its mistakes. In this case, the system is a machine learning model used for anomaly detection.

- Now, when this model is being used in real-world applications, the goal is to keep it updated with new information about where it made mistakes, like falsely identifying something as abnormal (false positives) or missing an actual abnormal event (false negatives).

- Here's where the unlearning framework comes in. The basic idea is, if we discover that a particular event (let's call it xt) was incorrectly identified as normal when it's actually abnormal (a false negative), we don't want the model to keep making the same mistake. So, instead of trying to maximize the probability that xt is normal given the previous data (x1...xt−1), we want to minimize this probability.

- However, doing this directly can cause some problems, like the model becoming too sensitive and forgetting everything it learned before. The unlearning approach is like a smart way of implementing this idea without causing new problems or hurting the model's performance on other data.

- It's essential to note that handling false positives (events that are mistakenly identified as abnormal) is relatively straightforward. In the paper, they focus more on explaining the challenges and techniques related to handling false negatives because these are the tricky cases.

## Paper 8: (importance, introduction, challenges of mul)
## Title: Machine Unlearning: Its Nature, Scope, and Importance for a “Delete Culture”

cite: Floridi, L. (2023). Machine Unlearning: its nature, scope, and importance for a “delete culture”. Philosophy & Technology, 36(2), 42.

link: https://link.springer.com/article/10.1007/s13347-023-00644-5

### Premise of the paper:
- **Accumulation of Digital vs. Analog Information:**<br>
The paper discusses how digital information can accumulate over time, unlike analog information. While digital data can be stored and added to continuously, analog information, like traditional documents or physical records, doesn't have the same easy accumulation.

- **Newborn Deletion Culture:**<br>
The concept of a "newborn deletion culture" is introduced. This culture addresses what information should be made entirely unavailable or, at the very least, practically inaccessible. The paper emphasizes the importance of considering the feasibility of making information unavailable before imposing it as an obligation. It highlights the ethical principle that what ought to be done must be possible to do.

- **Legal Content Removal in a Delete Culture:**<br>
In a "delete culture," legal content, not prohibited for reasons like being pornographic or violent, is removed mainly due to concerns related to intellectual property (IP) or privacy infringement. The paper notes that encryption and other practices making information inaccessible become more common in a delete culture, primarily due to these concerns.

- **Risks in the Development of Delete Culture:**<br>
The paper discusses the potential risks when transitioning to a delete culture, emphasizing that it's different from a cancel culture. It suggests that relying on conceptual tools developed in a recording culture (a culture that emphasizes preserving information) may not be sufficient. New approaches may be necessary to address issues like privacy and intellectual property infringement in the context of a delete culture

### We can't just make the data or information unavailable. The nuances of making the information unavailable.

- Cost and Reversibility of Blocking vs. Removing Information:<br>
  - The paper points out that preventing access to information (blocking) is more cost-effective and reversible compared to completely removing it. Blocking is simpler and can be undone more easily.

- Examples Illustrating the Challenges:

  - Cambridge Analytica Case:
    - After the Cambridge Analytica scandal, authorities required the company to delete the data it held. However, the paper notes that making information permanently unavailable is challenging. People might not fully comply, keeping copies or not genuinely deleting the data. Right to be Forgotten Case - Google Spain v AEPD and Mario Costeja González:
The landmark decision in 2014 mandated Google to exclude certain information from search results. However, this didn't involve deleting the information; instead, it made the information inaccessible in specific regions (e.g., EU). The paper highlights that determined individuals can still access it through methods like VPNs.

- Availability, Inaccessibility, and Circumstances:

Availability of Information
Inaccessibility of the Same Information
Circumstances under Which the Information is Unavailable
It uses these variables to describe situations where information is blocked or made inaccessible, emphasizing the challenges in achieving full and permanent removal.

- Global Access Challenges:
  - The text mentions that determined individuals can find ways to access information despite attempts to block or restrict it. This applies not only to public information but also to potentially sensitive or "secret" information. The strategy often revolves around making access difficult rather than impossible.

### Challenges of the paper:
- Problem in Machine Learning After Training:<br>The paper raises a question about what to do when a machine learning model, after being trained, poses privacy or intellectual property issues. It lists three potential solutions: (a) delete the model, (b) remove unwanted data and retrain the model from scratch (referred to as exact unlearning), or (c) "block" the information that the model has learned.

- How Machine Unlearning (MU) Achieves Blocking:<br>The paper explains how MU could achieve a form of blocking. For instance, if a chatbot like ChatGPT has information that it should not provide, MU could be used to make the chatbot "unlearn" that specific information. This means the information becomes unavailable in principle, not just inaccessible in practice. This approach aligns with concepts like the "right to be unlearnt," similar to the "right to be forgotten."

  - Example: If someone asks ChatGPT about sensitive financial information, MU could be used to make ChatGPT forget or not know the answer.

- Current Status of Machine Unlearning (MU):<br>
  - Challenges: MU is in its early stages and faces challenges like computational efficiency, scalability, technical reliability, and potential malicious attacks.

  - Pre-Unlearning Strategy: MU works better when the machine learning model is designed with unlearning in mind from the beginning. This is referred to as a "pre-unlearning" strategy, and it involves structuring the learning process to distinguish between what needs to be unlearned and what doesn't.

    - Example: Designing an ML model to easily unlearn specific types of information.

  - Algorithmic Level Definition: Some argue that MU is well-defined only at the algorithmic level, suggesting that its application might be clearer at the level of defining algorithms.

  - Debate in the AI Community: Despite ongoing debates, discussions about MU as a solution were notably absent in recent discussions related to decisions by authorities, such as the Italian Data Protection Authority's decision to block ChatGPT.

### Reasons for the Scope of Machine Unlearning:
The paper suggests that the field of MU is likely to gain momentum due to increasing legal and social pressures. Mentioned regulations like the AI Act and the AI Liability Directive indicate a growing awareness and regulatory push for responsible AI practices.

### The paper concludes with discussing how MUL can be misused, overused, underused:

- **Misuse of Machine Unlearning (MU):**<br>
  - Problem: MU, when in the wrong hands, can be misused. Imagine someone taking a trained model that provides information deemed undesirable or unacceptable and tweaking it to produce a revised model that no longer provides that information.
  - Potential Issues: In the hands of malicious individuals, MU could become a potent tool for censorship, spreading misinformation, propaganda, cyber-attacks, or even new forms of ransomware. This could lead to the removal of acceptable and desirable information, akin to a massive erasure of information reminiscent of dystopian scenarios like in the book "1984."
  - Challenges: Ethical concerns arise, especially in the context of training chatbots like ChatGPT. Decisions about what information to include or omit become complex in a world where managing vast amounts of information efficiently relies on machine learning tools.

- **Overuse of Machine Unlearning:**<br>
  - Risk: If MU methods become too successful, there's a risk of overuse. If it becomes easy and inexpensive to unlearn, there might be a tendency for individuals or organizations to take more risks during the initial training of their machine learning models.
  - Potential Issue: This might lead to a mindset of "collect first, unlearn later if needed," adopting an amoral, cost–benefit analysis approach, especially when dealing with privacy-related issues such as the right to be forgotten.

- **Underuse of Machine Unlearning:**<br>
  - Different Approaches: Removing and blocking information are two different approaches, each with its own costs, difficulties, and flexibility. Blocking information, a form of "machine silencing," could become a strong competitor to MU unless the law mandates the use of MU for specific cases.
  - Potential Debate: In the future, there might be a debate about whether unlearning or blocking is the right approach for highly sensitive information. Legal requirements may influence this choice.

### The Future of Machine Unlearning:
- Development: It has taken a long time to develop a culture of recording information. While developing a culture of deleting will not happen overnight, it's suggested that MU will likely be part of this developmental process.
Investment: The paper concludes by emphasizing the importance of investing in the development and study of MU as it plays a role in shaping the future of information management and ethical AI practices.

## Paper 9: (introduction, overview of mul)
## Title: “Amnesia” - A Selection of Machine Learning Models That Can Forget User Data Very Fast

cite: Schelter, S. (2020). Amnesia-a selection of machine learning models that can forget user data very fast. suicide, 8364(44035), 46992.

link: https://www.cidrdb.org/cidr2020/papers/p32-schelter-cidr20.pdf

### Introduction:
- same as other different papers, it first talks about the need of having the machine learning models' data be deleted. for like security and privacy issues. it stresses on "decremental learning"
- it talks about how the "right to be forgotten" is the legal necessity of any machine learning model, which is of the ethical concern.
- the word ***"decremental"*** is often used in the paper, it does not have any special and estabilished meaning, it can safely be assumed that the authors of this paper have coined this term, which in our case can be substituted with unlearning.
- So, in the context of this paper, "decremental update" essentially refers to a method of updating a machine learning model in a way that reduces or eliminates information related to a specific user, providing a way to forget or remove user-specific data from the model efficiently.
- also mentions, that it is easier to delete the data from the databases but not from the model that has already been trained. which infers the complications that comes with handling the data in machine learning models.

### An example for why just deleting the data from the dataset wont suffice in completely removing the data from a trained machine learning model

- Imagine there's a dataset about book ratings, and one user (let's call them User 51881) has read books related to teenage suicide. This information is sensitive, and the user wants it removed from the database under privacy regulations like the "right-to-be-forgotten."

- Now, the problem arises because there might be machine learning models, like a book recommendation system, that learned from the database before the user's records were deleted. For instance, the system might have a similarity matrix, showing how similar users are based on their reading history.

- In this case, even if the sensitive data about User 51881 is deleted, the model can still make educated guesses about their deleted reading history. For example, the system might find other users highly similar to User 51881 and show that these similar users also liked books about teenage suicide.

- So, despite the user's data being deleted, the model can still indirectly reveal their sensitive reading history by finding similarities with other users who liked similar books. This poses a potential privacy risk, showing that simply removing data from a database might not be enough, as machine learning models can still make connections and predictions based on previously learned patterns

### Incremental model and decremental model: 

1. Incremental Model:
  - Definition: Incremental models involve adding new data to the existing dataset, updating the model based on the additional information.<br>
  - Example: If you have a recommendation system and a new user rates a book, the system incrementally updates its understanding to provide better recommendations for that user.

2. Decremental Model:
  - Definition: Decremental models, on the other hand, focus on removing data from the existing dataset, adjusting the model by forgetting or unlearning certain information.<br>
  - Example: If a user requests their data to be removed from a recommendation system (for privacy reasons), the system should be able to decrementally update itself to forget that user's preferences.

3. Tasks: ( these are the different tasks that this paper works on, have not done a deep analysis on these, just to give an idea ) 

  - Item-Based Collaborative Filtering (Recommender Systems):
    - Explanation: This is a type of recommendation system where items are recommended based on the preferences and behavior of users.
    - Example: If User A and User B like similar books, and User A likes a new book, the system incrementally learns that User B might also like the new book.

  - Ridge Regression (Regression):
    - Explanation: Ridge Regression is a regression technique that prevents overfitting by adding a penalty term to the linear regression equation.
    - Example: In a housing price prediction model, if new data about house prices is added, the system incrementally adjusts the regression model to maintain accuracy.

  - k-Nearest Neighbors with Locality Sensitive Hashing (Classification):
    - Explanation: k-Nearest Neighbors is a classification algorithm that classifies a data point based on the majority class of its neighbors. Locality Sensitive Hashing helps speed up the process.
    - Example: In a spam email classifier, if new emails are added, the system incrementally updates its understanding of what features are indicative of spam or not.

  - Multinomial Naive Bayes (Classification):
    - Explanation: Multinomial Naive Bayes is a classification algorithm based on Bayes' theorem, particularly suitable for discrete data, such as text classification where the data can be represented as word counts in a document.
    - Example: In a spam email classifier using Multinomial Naive Bayes, if new emails are added, the system incrementally adjusts its understanding of word frequencies associated with spam or non-spam emails. For example, if the word "free" appears frequently in new spam emails, the system incrementally updates its model to better recognize this pattern.

### to conclude:
- this paper gives a brief idea on how to deal with data with machine learning models, in this context it mostly falls in our field of research too. on the other hand the gist of the paper is totally different and mainly deals with something different.

## Paper 10: ( applications of mul)
## Title: Efficient Repair of Polluted Machine Learning Systems via Causal Unlearning

cite: Yinzhi Cao, Alexander Fangxiao Yu, Andrew Aday, Eric Stahl, Jon Merwine, and Junfeng Yang. 2018. Efficient Repair of Polluted Machine Learning Systems via Causal Unlearning. In Proceedings of 2018 ACM Asia Conference on Computer and Communications Security, Incheon, Republic of Korea, June 4–8, 2018 (ASIA CCS ’18), 13 pages.

link: https://dl.acm.org/doi/abs/10.1145/3196494.3196517

### Introduction: 

- Imagine you have a smart system, like those used in everyday applications. These systems learn from examples but can be tricked by malicious attacks, especially one called "data pollution." This attack involves sneaking in misleading training data, which makes the system learn wrong things and give incorrect results.

- Now, fixing this problem is usually done by finding and removing the bad data and re-teaching the system. However, these systems often deal with massive amounts of data (like millions of examples), making it impossible for a person to manually check each one for problems.

- This paper introduces a solution called "causal unlearning" and a system named "Karma." What they do is pretty cool. They automatically figure out which data is causing the issue and needs to be removed. This saves a lot of time for the people managing these systems because they don't have to manually inspect everything.

- The paper tested Karma on three different learning systems, and the results were great. It made fixing these systems much easier, and it was really good at finding the problematic data. So, in simple terms, Karma is like a smart helper that quickly finds and fixes mistakes in these smart systems.

### How does **KARMA** work: 
- Imagine you have a smart system that sometimes makes mistakes because someone played a trick by giving it wrong examples during its learning phase. Karma is like a tool that comes to the rescue to fix these mistakes.

- Here's how Karma works:
  - Trace the Trouble: When someone tricks the system with bad examples, they leave a trail of problems. Karma follows this trail, starting from the wrong examples, going to the messed-up learning model, and ending with the errors in the results.

  - Smart Search: Karma searches through different groups of examples that were used to teach the system. It figures out which group is causing the most mistakes in the results.

  - Trial and Error: Karma does a bit of experimenting. It removes a group of examples that seem to be causing trouble, creates a new learning model without those examples, and checks if the mistakes in the results are fixed.

  - Two-Step Cleanup: Karma makes it easy for the people managing the system. First, it assumes that some users will report mistakes in the system's results. Second, it relies on these users' reports to gradually clean up the system. So, you don't have to find all the mistakes at once.

  - Admin Check: Karma also needs a bit of help from administrators. It points out which examples might be causing trouble, and administrators check to make sure Karma is right.

  - Not Perfect, but Helpful: Karma might sometimes say an example is bad when it's not, or it might miss a bad example. But that's okay because it's better to be safe. It's like Karma is looking for weird things in the examples to make sure the system works better.

### to conclude: 
- this paper is on an application or framework which is based off machine unlearning, this framework is called karma. this framework detects any bad data and tries to clean it.
- we can use this for when we talk about the applications of machine unlearning.

## Paper 11: ( method of mul)
## Title: When Machine Unlearning Jeopardizes Privacy
Cite: Chen, Min, et al. "When machine unlearning jeopardizes privacy." Proceedings of the 2021 ACM SIGSAC conference on computer and communications security. 2021. 
arXiv:2005.02205 [cs.CR]


## •	Introduction
In this paper, we study to what extent data is indelibly imprinted in an ML model by quantifying the additional information leakage caused by machine unlearning.
The most legitimate way to implement machine unlearning is to remove the data sample requested to be deleted (referred to as target sample), and retrain the ML model from scratch, but this incurs high computational overhead.
Machine unlearning naturally generates two versions of ML models, namely the original model and the unlearned model, and creates a discrepancy between them due to the target sample’s deletion. It can be that machine unlearning may leave some imprint of the data deleted, and thus create unintended privacy risks.

## •	Machine Learning and Unlearning
Machine learning classification is the most common ML task. An ML classifier M maps a data sample 𝑥 to posterior probabilities P, where P is a vector of entries indicating the probability of 𝑥 belonging to a specific class 𝑦 according to the model M. The sum of all values in P is 1 by definition. To construct an ML model, one needs to collect a set of data samples, referred to as the training set D. The model owner should remove the target sample 𝑥 from its training set D. Moreover, any influence of 𝑥 on the model M should also be removed. This process is referred to as machine unlearning.

## •	Retraining from Scratch
The most legitimate way to implement machine unlearning is to retrain the whole ML model from scratch. Formally, denoting the original model as M𝑜 and its training dataset as D , this approach consists of training a new model M𝑢 on dataset D𝑢 = D𝑜 \ 𝑥pow2 We call this M𝑢 the unlearned model.
Drawback: Retraining from scratch is easy to implement. However, when the size of the original dataset D𝑜 is large and the model is complex, the computational overhead of retraining is too large.

## •	General method to implement machine unlearning
## o	SISA.
SISA works in an ensemble style, which is an efficient and general method to implement machine unlearning. The training dataset D𝑜 in SISA is partitioned into 𝑘 disjoint parts Do1, Do2, • • • ,Do𝑘. The model owner trains a set of original ML models Mo1,Mo2, • • • ,Mo𝑘 on each corresponding dataset Do𝑖 . When the model owner receives a request to delete a data sample 𝑥, it just needs to retrain the sub-model Mo  that contains 𝑥, resulting in unlearned model Mu𝑖. Sub-models that do not contain 𝑥 remain unchanged. Notice that the size of dataset Do𝑖 is much smaller than D ; thus, the computational overhead of SISA is much smaller than the “retraining from scratch” method.


## •	Adversary’s Goal
Given a target sample 𝑥, an original model, and its unlearned model, the adversary aims to infer whether 𝑥 is unlearned from the original model.

## •	PRIVACY DEGRADATION MEASUREMENT

o	Degradation Count (DegCount) : It calculates the proportion of target samples whose true membership status is predicted with higher confidence by our attack than by classical membership inference

o	Degradation Rate (DegRate) : It calculates the average confidence improvement rate of our attack predicting the true membership status compared to classical membership inference

## •	EVALUATION

o	first conduct an end-to-end experiment to validate the effectiveness of our attack on multiple datasets using the most straightforward unlearning method, i.e., retraining from scratch.

o	Second, we compare different feature construction methods and provide a summary of the most appropriate to choose depending on the context

o	Third, we evaluate the impact of over fitting and of different hyper parameters

o	Fourth, we conduct experiments to evaluate dataset and model transferability between shadow model and target model

o	Finally, compare the effectiveness of the attack against the SISA unlearning method

## •	Datasets
 We run experiments on two different types of datasets: categorical datasets and image datasets. 
o	The categorical datasets are used to evaluate the vulnerability of simple machine learning models. 
o	The image datasets are used to evaluate the vulnerability of the convolutional neural networks.

## •	MEMBERSHIP INFERENCE IN MACHINE UNLEARNING
o	Attack Pipeline
It consists of three phases: posteriors generation, feature construction and membership inference.

• Posteriors Generation
The adversary has access to two versions of the target ML model, the original model M𝑜 and the unlearned model M𝑢. Given a target sample 𝑥, the adversary queries M𝑜 and M𝑢, and obtains the corresponding posteriors, i.e., P𝑜 and P𝑢.

•	Feature Construction
Given the two posteriors P𝑜 and P𝑢, the adversary aggregates them to construct the feature vector F.

•	Inference
The adversary sends the obtained F to the attack model, which is a binary classifier, to determine whether the target sample 𝑥 is in the training set of the original model.

o	Attack Model Training
First assume the adversary has a local dataset, which can be called the shadow dataset D𝑠. The shadow dataset can come from a different distribution than the one used to train the target model. To infer whether the target sample 𝑥 is in the original model or not, the core idea is to train an attack model M𝐴 that captures the difference between the two posteriors. The main is that, if the target sample 𝑥 is deleted and the two models M𝑜 and M𝑢 will behave differently.

o	Training Shadow Models
To mimic the behavior of the target model, the adversary needs to train a shadow original model and a set of shadow unlearned models. To do this, the adversary first partitions D𝑠 into two disjoint parts, the shadow negative set D𝑠𝑛 and the shadow positive set D𝑠𝑝. The shadow positive set D𝑠𝑝 is used to train the shadow original model M𝑠𝑜. The shadow unlearned model M𝑠𝑢 is trained by deleting samples from D𝑠𝑝.


//this is a reference understanding 
## •	Membership Inference 
Shokri et al. presented the first membership inference attack against ML models. The main idea is to use shadow models to mimic the target model’s behavior to generate training data for the attack model. Salem et al. [60] gradually removed the assumptions of [64] by proposing three different attack methods. Since then, membership inference has been extensively investigated in various ML models and tasks, such as federated learning [46], white-box classification [48], generative adversarial networks [13, 28], natural language processing [67], and computer vision segmentation [30]

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership Inference Attacks Against Machine Learning Models. In IEEE Symposium on Security and Privacy (S&P), pages 3–18. IEEE, 2017.


## Paper 11: ( overview of mul)

## Title: Towards making systems forget with machine unlearning

Cite: Y. Cao and J. Yang, "Towards Making Systems Forget with Machine Unlearning," 2015 IEEE Symposium on Security and Privacy, San Jose, CA, USA, 2015, pp. 463-480, doi: 10.1109/SP.2015.35.

## •	Evaluation metric

The usefulness of forgetting systems can be evaluated with two metrics: how completely they can forget data (completeness) and how quickly they can do so (timeliness). The higher these metrics, the better the systems are at restoring privacy, security, and usability.

## •	Approach

To prepare for unlearning, we transform learning algorithms in a system to a form consisting of a small number of summations. 
Each summation is the sum of some efficiently computable transformation of the training data samples. 
The learning algorithms depend only on the summations, not individual data. 
These summations are saved together with the trained model. 
Then, in the unlearning process, we subtract the data to forget from each summation, and then update the model. 
As a result the time required is less and faster retraining from scratch.
They propose that a general efficient unlearning approach applicable to any algorithm that can be converted to the summation form.

## •	This paper makes four main contributions:

o	The concept of forgetting systems that restore privacy, security, and usability by forgetting data lineage completely and quickly;

o	 A general unlearning approach that converts learning algorithms into a summation form for efficiently forgetting data lineage;

o	 An evaluation of our approach on real-world systems/algorithms demonstrating that it is practical, complete, fast, and easy to use; and

o	 The practical data pollution attacks we created against real-world systems/algorithms.


## •	machine learning system has three processing stages.

o	 Feature selection: During this stage, the system selects, from all features of the training data, a set of features most crucial for classifying data.

o	Model training: The system extracts the values of the selected features from each training data sample into a feature vector. It feeds the feature vectors and the malicious or benign labels of all training data samples into some machine learning algorithm to construct a succinct model.

o	Prediction: When the system receives an unknown data sample, it extracts the sample’s feature vector and uses the model to predict whether the sample is malicious or benign

## ///Paper still in Progress
