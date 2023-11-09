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

# Paper3:
Author -- Yashwanth R  @https://github.com/ShiroganeMiyuki-0
## Title: Coded Machine Unlearning
- Coded machine unlearning: A new framework for removing data from trained ML models using data encoding and ensemble learning. It aims to achieve perfect unlearning with better performance and lower cost than uncoded methods.
- Problem setup: A regression problem with a training dataset of n samples and d features. The loss function is regularized and kernelized. The model uses random projections to reduce the dimensionality of the features. The unlearning protocol must satisfy the perfect unlearning criterion.
- Learning protocol: The model uses a linear encoder to transform the training dataset into r coded shards, each with n/s samples. The shards are assigned to r weak learners that are trained independently and aggregated using an averaging function. The encoder usa random binary matrix with rate τ = n/m and density ρ.
- Unlearning protocol: The model identifies the coded shards that contain the sample to be unlearned and updates them by subtracting the sample from the corresponding coded samples. Then, the model retrains the affected weak learners and updates the aggregate model. The protocol guarantees perfect unlearning.
- Synthetic data experiments: The authors test their coded learning and unlearning protocol on three synthetic datasets with different features and response variables. They show that coding provides a better trade-off between performance and unlearning cost for datasets with heavy-tailed features, but not for datasets with normal features.
- Conclusion: The authors summarize their main contributions and discuss some possible directions for future work, such as extending the protocol to other models, studying different classes of codes, and exploring the role of influential samples in coded learning and unlearning.
