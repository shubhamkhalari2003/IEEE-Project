Real-Time Driver Alertness Monitoring with
Optimized Deep Learning and Haar-Cascade
Methods
Karan Singh Rawat
Department of Computing
Technologies, School of Computing
SRM Institute of Science and
Technology, Kattankulathur
Chengalpattu, India, 603203
kk8873@srmist.edu.in
Shubham Kumar
Department of Computing
Technologies, School of Computing
SRM Institute of Science and
Technology, Kattankulathur
Chengalpattu, India, 603203
ss5552@srmist.edu.in
M. Kowsigan
Department of Computing
Technologies, School of Computing
SRM Institute of Science and
Technology, Kattankulathur
Chengalpattu, India, 603203
kowsigam@srmist.edu.in
Ramamoorthy S
Department of Computing
Technologies, School of Computing
SRM Institute of Science and
Technology, Kattankulathur
Chengalpattu, India, 603203
ramamoos@srmist.edu.in
Madhavan P
Department of Computing
Technologies, School of Computing
SRM Institute of Science and
Technology, Kattankulathur
Chengalpattu, India, 603203
madhavap@srmist.edu.in
Abstract— Driver fatigue is a leading cause of road accidents.
This research presents a real-time driver drowsiness
detection system using deep learning, optimized for lowpower
embedded systems. By compressing a complex model
into a lightweight version, the system achieves an 85.5%
accuracy in detecting various alertness states while
maintaining energy efficiency. Utilizing facial landmark
tracking and the Haar-Cascade method, the approach
ensures fast and accurate detection, making it ideal for realtime
vehicle safety applications. The system's integration into
modern vehicles promises enhanced driver fatigue
management and accident prevention.
Keywords: Drowsiness Detection, Low-Power Systems,
Facial Landmark Tracking, Real-Time Safety, Neural
Network Optimization.
I. INTRODUCTION
Driver drowsiness is a leading factor in many motor vehicle
accidents, with extensive research and data highlighting its
significant impact. For example, in 2014, the National Highway
Traffic Safety Administration (NHTSA) recorded 846 fatalities
directly attributed to drowsy driving — a number that has shown
little change over the past decade. Additionally, between 2005
and 2009, drowsy driving was estimated to be responsible for an
average of 83,000 crashes each year, demonstrating the
persistent threat posed by fatigued drivers. These alarming
statistics emphasize the critical need for an effective system to
detect and reduce the risks associated with driver drowsiness[1].
A robust drowsiness detection system must be able to accurately
identify signs of fatigue in real-time, enabling timely
interventions. Such a system could range from issuing a direct
alert to the driver to taking more advanced actions, such as
transferring control to an autonomous vehicle system, depending
on the vehicle's capabilities.
Researchers have explored various methods for detecting driver
drowsiness, which can generally be classified into three broad
categories. The first involves analyzing the driving patterns,
including advanced techniques that monitor specific behaviors
such as steering wheel movements, acceleration and deceleration
patterns, and lane deviations. While these behaviors can indicate
the onset of drowsiness, their effectiveness often depends on
specific driving conditions, making them less universally
applicable[2].
The second category focuses on psychophysiological signal
assessment, which involves monitoring the driver’s biosignals,
such as EEG (Electroencephalography), ECG
(Electrocardiography), and EOG (Electrooculogram). These
signals can provide detailed insights into the driver’s level of
alertness. However, the practical challenges of these methods,
especially the discomfort and intrusiveness of wearing sensors to
capture these signals, limit their feasibility for widespread use in
real-world driving scenarios[3].
Given the limitations of the first two approaches, computer
vision-based methods have gained significant traction in driver
monitoring. These methods focus on detecting visual indicators of
drowsiness, such as eye closure, yawning, changes in facial
expressions, and head movements. Computer vision offers a nonintrusive
solution that can be implemented in real-time, making
it an increasingly preferred choice for integrating drowsiness
detection into modern vehicles. This approach not only enhances
the accuracy of drowsiness detection but also ensures the system
remains user-friendly and comfortable for drivers, promoting
safer driving conditions without imposing additional burdens[4].
The development and refinement of such systems are essential in
reducing the risk of accidents caused by driver fatigue, thereby
contributing to overall road safety. As technology continues to
advance, the integration of these systems into everyday vehicles
will become increasingly sophisticated, offering even greater
protection for drivers and passengers alike[5].
This research introduces an innovative approach to real-time
drowsiness detection using a deep learning model enhanced by
computer vision techniques. The system carefully analyzes key
facial features of the driver to accurately classify their state into
three distinct categories: normal, drowsy, and asleep. One of the
standout features of the proposed model is its ability to undergo
significant compression without a loss in accuracy, making it
ideally suited for deployment on cost-effective, resource-limited
embedded systems. This design ensures that the drowsiness
detection system can be seamlessly integrated into standard
vehicle electronics, such as the Electronic Control Unit (ECU),
while maintaining real-time operation and low power
consumption. This approach not only enhances the feasibility of
widespread adoption in everyday vehicles but also provides a
robust, energy-efficient solution that can significantly contribute
to road safety by effectively monitoring and addressing driver
fatigue[6].
II. LITERATURE SURVEY
Driver drowsiness detection has become essential in reducing
traffic accidents caused by driver fatigue. This survey
synthesizes key advancements, focusing on vehicle-based
detection methods, physiological signal monitoring, computer
vision techniques, embedded system integration, and multimodal
approaches.
Vehicle-Based Drowsiness Detection:
Early approaches in drowsiness detection focused on analyzing
driving behavior such as steering variability, lane position, and
braking patterns. Wierwille et al. (1994) observed that increased
variability in steering correlated with driver fatigue, and Dinges
et al. (1998) noted a connection between lane departures and
reduced alertness. Zhu and Ji (2004) [2] developed a nonintrusive
method to detect fatigue by monitoring lane position
and steering angles in real time. Ji et al. (2004) [4] extended this
by including predictive analysis of driver fatigue, improving
system responsiveness. Despite these contributions, vehiclebased
systems struggle with reliability in varying environments,
as Might and Baldwin (2009) demonstrated, due to factors such
as road conditions and vehicle speed. To overcome these
limitations, Wang and Zhang (2021) [8] suggested incorporating
environmental and physiological data to reduce false positives
and missed detections.
Physiological Signal Monitoring:
Monitoring physiological signals, such as EEG, ECG, and EOG,
directly measures the driver’s internal state, providing valuable
insights into fatigue. Patel et al. (2011) [1] explored driver fatigue
detection through neural network analysis of heart rate variability,
establishing that heart rate patterns can reliably indicate fatigue.
Akerstedt et al. (2003) and Huang et al. (2016) expanded on this
by examining EEG frequency bands and ECG readings,
respectively, with Huang et al. linking heart rate variability to
fatigue levels. Additionally, Johns et al. (2002) utilized EOG to
monitor eyelid movement, finding prolonged closures indicative
of drowsiness. While these approaches offer precision, there are
practical challenges, as sensors can be intrusive and may yield
erroneous readings due to external stressors. Roberts and Adams
(2024) [17] suggested hybrid approaches combining biometric
sensors with machine learning for enhanced monitoring of
alertness.
Computer Vision-Based Methods:
The development of computer vision techniques has opened new
avenues for drowsiness detection by focusing on facial
expressions and head movements. Viola and Jones (2001) laid the
groundwork for real-time face detection algorithms, which
Bergasa et al. (2006) adapted to monitor driver drowsiness
through eyelid and gaze tracking. This approach was further
improved by Mao and Zhou (2018) [7], who used deep learning
and computer vision to achieve more accurate results.
Abdusalomov et al. (2023) [5] introduced eye-blink analysis to
improve road safety, while Park et al. (2017) employed CNNs to
identify drowsiness through eye and mouth movements. Though
effective, these deep learning models can be resource-intensive;
Hinton et al. (2015) addressed this issue through model
compression, particularly knowledge distillation, allowing realtime
use on embedded systems. Kumar and Yadav (2020) [10]
leveraged similar techniques for real-time facial emotion
recognition in drowsiness detection, enhancing system efficiency.
Embedded Systems and Model Compression:
The increasing integration of advanced driver-assistance systems
(ADAS) requires drowsiness detection systems to operate
efficiently on embedded platforms. Lin et al. (2018) developed a
lightweight deep learning model for the Jetson TK1 platform,
achieving high accuracy with real-time performance. Ding and
Wu (2019) [11] emphasized the importance of model
compression techniques, such as pruning and quantization, to
optimize these models for embedded deployment, balancing
power efficiency and processing speed. Dewi et al. (2022) [16]
further explored eye aspect ratios for real-time detection, showing
promising results in embedded environments.
Multimodal Approaches and Data Fusion:
Recent studies advocate for multimodal approaches that integrate
video analysis, physiological data, and vehicle data, enhancing the
reliability of drowsiness detection. Khan and Lee (2020) [12]
combined CNN models with multimodal inputs to improve
accuracy and robustness. Alonso and Suárez (2021) [13] provided
a comprehensive survey on multimodal detection systems,
highlighting the strengths of data fusion in achieving higher
reliability. Multimodal systems are particularly promising, as they
offer redundancy and robustness, mitigating the weaknesses of
any single detection method. Roberts and Adams (2024) [17]
demonstrated the effectiveness of combining biometric sensors
with machine learning techniques to monitor driver alertness in
real time.
III. EXISTING SYSTEM
In the realm of driver drowsiness detection, numerous systems
have been developed, each employing different methodologies
to identify fatigue in drivers. These systems leverage various
data sources, including vehicle dynamics, physiological signals,
and visual cues, to assess the driver’s level of alertness[13].
Patel et al. [1] present a framework for driver recognition through
neural network analysis of heart rate variability (HRV) data.
Their approach focuses on real-time analysis and trains neurons
to detect fatigue-related changes in HRV patterns. The system
was able to detect the sleeping conditions accurately, although
specific performance metrics such as accuracy, recall and F1-
score were not detailed in the findings but their method showed
great potential to reduce the risk of accidents by providing an
effective fatigue detection method[14].
Ahmed et al. [5] focus on the use of deep learning techniques to
detect driver sleep. Their method uses a convolutional neural
network (CNN) to analyze facial expression and eye movements
and classify driver state and analysis sleep and warning
categories. The developed CNN model achieved a classification
accuracy of 91.2%, outperforming traditional devices learning
methods. This approach effectively solves the challenge of
manual sleep recognition by automating the process, resulting in
a more reliable solution in real time and it will enhance road
safety more efficiently[15].
Figure .1. driver Drowsiness Detection
Abdusalomov et al. [5] present a creative model integrating both
computer vision and deep learning methods for real-time sleep
detection. While the study addresses numerous use cases, its
target application is focused on improving road safety through
eye-tracking technology. Furthermore, they discuss the benefits
of machine learning in terms of improving performance with
driver fatigue detection. On consideration of recent strides in
deep learning algorithms applied to computer vision as a
significant advancement which may help reduce accidents and
driver fatigue[16].
Figure 2. Accuracy Metrics for Drowsiness Detection
Model
Figure 2 exhibits the accuracy evaluation of the drowsiness
detection model investigated by Abdusalomov et al. (2023) in
relation to their utilization of Deep Learning to detect drowsiness
in the Graduate Student Thesis "Real-Time Deep Learning-Based
Drowsiness Detection". It shows the model's effects on its various
detection types, including:
 Drowsy Eyes Detection: Indicates the accuracy of the
model in detecting drowsy eyes at 95.8%.
 Open Eyes Detection: Illustrates the accuracy of detecting
open eyes, which appears at 97%.
 Yawning Detection: Shows the model's accuracy in
identifying yawning at 0.84%.
The model clearly demonstrates that it is the most accurate in
detecting drowsy and open eyes, which is relatively important if
the goal of the model is for real-time applications involving driver
alertness. It can be noted, however, that the yawning detection
accuracy is an area for improvement.
In total, the potential improvements that come with the method of
employing today's advances in Deep Learning and Computer
Vision technologies to enhance road safety is a main theme in
these findings. The advances in education to use in drivers with
methods using these methods have the potential to make driver
monitoring systems more efficient, which could considerably
enhance safety not only in terms of vehicle operations, but also in
terms of more general usage in effective Intelligent
Transportation Systems (ITS)[17].
Figure 1. Overview of Driver Drowsiness Detection
This figure provides a visual overview of the driver drowsiness
detection methodology based on deep learning and computer
vision approaches. It visually depicts the sensing of eye
movement tracking, yawning detection, or head tilt to detect the
driver’s alertness in real-time.
Figure 2. Drowsiness Detection Model Accuracy Metrics
This figure visually compares analysis and accuracy of the
drowsiness detection model in recognizing and detecting different
states of driver alertness, such as drowsy eyes, open eyes, the
presence of yawning, and left side leaning. The model shows
significant strength for detecting open or drowsy eyes while
yawning detection accuracy should improve.
IV. PROPOSED SYSTEM
The outlined driver drowsiness detection system brings together
advanced image-processing techniques and machine learning
techniques to bring down the limits of existing systems. The
system emphasizes real-time, non-intrusive analysis of facial
cues to develop an accurate solution that is user-friendly and
provides capabilities to detect driver fatigue.
A. Image Acquisition
 Dataset: The system uses a diverse set of images and
videos of drivers in various lighting, head positions, and
facial expressions.
 Capture Conditions:Data is gathered from multiple
driving scenarios, including nighttime, rain, and heavy
traffic, to improve robustness in real-world conditions.
B. Image Pre-processing
 Image Enhancement: The system uses methods including
histogram equalization and brightness adjustments enhance
facial features to make drowsiness indicators, such as eyelid
closure, more noticeable.
 Noise reduction: The system uses Gaussian and median
filters to add noise reduction to the visual data, particularly
when image quality may be compromised due to low-light
conditions that could impact the reliability of the
information being corrected.
C. Feature Extraction
 Eye Closure Detection: In fundamental features of
drowsiness, the system tracks a driver's eye closure rate by
calculating the percentage of closure present over a given
time.
 Yawning Detection: Tracking algorithms examine the
feature around the mouth to find frequency and intensity of
yawning behavior, which can signify fatigue in drivers.
D. Algorithm selection and training
 Convolutional Neural Network (CNN): This algorithm
uses CNN to extract and analyze facial features. These
neurons are particularly effective at recognizing patterns in
image data.
 Random forests: Random forest classifications are used to
describe the complex relationships between these
omissions. This group learning approach combines multiple
decision trees for more accurate predictions.
E. Drowsiness Classification
 State classification: The trained model classifies the
driver’s state into three categories: alert, drowsy, or asleep.
 Real-time feedback: When the system detects signs of
drowsiness, it immediately alerts the driver. This can also
include audio and visual warnings, such as beeping sounds
or flashing lights.
F. Output Generation
 Alert System: The system triggers audible and visual alerts
when drowsiness is detected. Additionally, it can send
notifications to connected devices (e.g., smartphones) or
activate vehicle safety systems.
 Intervention Recommendations: Depending on the
severity of detected drowsiness, the system can
recommend interventions, such as rest breaks. In
vehicles equipped with autonomous driving features, it
can prompt the activation of those systems.
G. System Evaluation
 Accuracy Assessment: The system’s performance is
continuously evaluated against ground truth data, ensuring
reliability and high accuracy in real-time scenarios under
varying driving conditions.
 System Optimization: Developments are made to the model
algorithms and thresholds in response to data collected and
performance trials, which contribute to minimizing false
positive detection and enhancing reliability.
Figure.3. Flow Graph of the Proposed System
H. Vehicle Integration with an Arduino System
The alert system model is further integrated with an Arduino
based system that enables drowsy driving speed control. A smart
system and motor will reduce the vehicle's speed once
drowsiness exceeds a limit set (threshold value). This smart
system will not only reduce the speed of the vehicle, but it will
also reduce the chance of drowsy driving incidents.
V.RESULTS AND DISSCUSSIONS
The integration of various analyses in our driver drowsiness
detection project provided significant insights into the
effectiveness of detecting driver fatigue. The results below
summarize key findings from the project:
 Eye Closure Analysis: This analysis classified the driver’s
alertness based on the percentage of eye closure detected
(refer to Figure (a)).
 Severely Drowsy: When the driver’s eye closure exceeded
80% for an extended period, it was classified as severely
drowsy, indicating that immediate action is needed.
Eye Aspect Ratio (EAR) Calculation
The Eye Aspect Ratio (EAR) is a key metric used to quantify eye
closure as an indicator of drowsiness. The EAR is calculated
based on the vertical and horizontal distances between specific
facial landmarks around the eye region. These landmarks are
detected using a facial landmark detection algorithm. The EAR
provides a robust, real-time measure of the driver’s eye state, and
is given by the following formula:
Where EARL and EARR represent the Eye Aspect Ratios for the
left and right eyes, respectively. The individual EAR for each eye
is computed as:
In this equation:
P1, P2, P3, P4, P5, and P6 are specific coordinates of key points
around the eye.
|P2 - P6| represents the distance between points P2 and P6, which
corresponds to the vertical eye opening.
|P3 - P5| represents the distance between points P3 and P5, which
also provides a vertical measurement.
|P1 - P4| represents the distance between points P1 and P4, which
corresponds to the horizontal width of the eye.
Yawning Frequency Analysis: This analysis monitored the
frequency of yawning as an indicator of fatigue.
 Frequent Yawning: A high frequency of yawning, defined as
more than 3 yawns within a 10-minute period, indicated
significant drowsiness.
 Occasional Yawning: Occasional yawning, defined as 1-2
yawns, suggested mild fatigue but not immediate danger.
Head Position Analysis: The head position was tracked to assess
driver alertness.
 Head Drooping: Continuous detection of head drooping
indicated severe drowsiness and necessitated intervention.
 Head Tilting: Analysis of head tilting showed patterns
associated with fatigue, enabling early warning alerts.
Machine Learning Integration: Following the preliminary
analyses, we applied Convolutional Neural Networks (CNN) to
enhance the detection model. By training the CNN on a dataset of
driver images with corresponding labels, we achieved an
impressive accuracy of 92%. This accuracy underscores the
effectiveness of using CNN model for identifying subtle fatigue
indicators that traditional methods might miss.
The combination of eye closure, yawning, and head position
analyses provided a robust evaluation framework for assessing
driver alertness. While these individual analyses laid a solid
groundwork, their limitations in handling complex behavioral
patterns were addressed through the integration of machine
learning. The CNN’s ability to learn from labeled data allowed it
to detect intricate patterns and correlations, significantly
improving prediction accuracy compared to traditional feature
extraction techniques. This advancement highlights the potential
of machine learning in enhancing drowsiness detection systems,
ultimately contributing to safer driving practices.
Figure 5: Result Analysis by the Proposed Approach
The final classification resulted in identifying the driver as
moderately drowsy, prompting the system to issue an alert for
corrective action. Figure 5 presents the result analysis,
demonstrating that the proposed approach effectively ensures that
drivers exhibiting signs of drowsiness are promptly notified,
enhancing overall road safety
Study Methodology &
Accuracy
Limitations
Proposed System CNN + Random
Forest, Haar
Cascade (92%)
Lower yawning
accuracy (84%)
Patel et al. (2011) [1] Neural Network
(HRV data);
Accuracy: N/A
No detailed
performance metrics
Ahmed et al. (2021)
[3]
CNN (facial
expression, 91.2%)
Limited real-world
driving data
Ji et al. (2004) [4] Predictive Modeling
(87%)
Lacks deep learning
Wang & Zhang
(2021) [8]
Hybrid DL + CNN
(90%)
High processing
time
Fig.6 Comparative Analysis of Drowsiness Detection Models
VI.CONCLUSION
Our research has successfully demonstrated the transformative
potential of machine learning and image processing techniques
in the realm of driver safety. By accurately detecting drowsiness
with a notable accuracy of 92%, the proposed framework
effectively addresses critical challenges associated with driver
fatigue.
The integration of advanced deep learning techniques for feature
extraction and real-time image analysis allows for precise and
reliable detection of drowsiness, significantly surpassing
traditional monitoring methods.
This framework offers several key advantages for drivers and
road safety stakeholders. Firstly, it reduces reliance on manual
checks, which can often be subjective and error-prone. Secondly,
the system processes large volumes of driver data quickly,
making it suitable for various driving environments.
Moreover, this research paves the way for future applications of
machine learning and image processing in diverse fields. The
principles established here could easily be adapted for
monitoring fatigue in other high-stakes environments, such as
aviation and industrial operations.
In conclusion, the proposed drowsiness detection system not
only advances the field of driver safety but also establishes a
foundation for future innovations in monitoring technologies.
Continued exploration and refinement of these techniques could
lead to even more accurate and efficient systems, ultimately
benefiting both drivers and the broader community.
REFERENCES
[1]. Patel, M., Lal, S., Kavanagh, D., & Rossiter, P. (2011). "Investigating Driver
Fatigue by Neural Network Analysis of Heart Rate Variability," Accident
Analysis & Prevention, 41(4), pp. 772-778.
[2]. Zhu, Z., Ji, Q. (2004). "Real-time Non-intrusive Monitoring of Driver Fatigue,"
in Proceedings of the 7th IEEE International Conference on Intelligent
Transportation Systems, pp. 657-662.
[3]. Ahmed, M. M., Khalil, M., Zeidan, E. (2021). "Driver Drowsiness Detection
Based on Deep Learning Techniques," IEEE Access, 9, pp. 112938-112946.
[4]. Ji, Q., Zhu, Z., Lan, P. (2004). "Real-time Non-intrusive Monitoring and
Predictive Analysis of Driver Fatigue," IEEE Transactions on Vehicular
Technology, 53(4), pp. 1052-1068.
[5]. Abdusalomov, A.B., Nasimov, R. Cho, Y. (2023). Real-Time Deep Learning-
Based Drowsiness Detection: Utilizing Computer-Vision and Eye-Blink Analysis
to Improve Road Safety, Sensors, 23(14), pp. 6459.
[6]. Dua, M., Shakshi, R. Singla, R., Jangra, A. (2021). Deep CNN Models Based
Ensemble Approach to Driver Drowsiness Detection. Neural Computing and
Applications, 33, pp. 3155-3168.
[7]. Mao, Q., & Zhou, J. (2018). "A novel real-time driver drowsiness detection
system based on deep learning and computer vision." IEEE Access, 6, 40715-
40724.
[8]. Wang, Y., & Zhang, Q. (2021). "Driver drowsiness detection using hybrid deep
learning and convolutional neural networks." Computers, Materials & Continua,
67(3), 2791-2804.
[9]. Zhang, Z., & Yang, X. (2019). "Facial landmark detection using a multi-task
deep learning framework." IEEE Transactions on Pattern Analysis and Machine
Intelligence, 41(6), 1373-1386.
[10]. Kumar, A., & Yadav, R. (2020). "Real-time facial emotion recognition for
driver drowsiness detection using deep convolutional neural networks."
Computers, 9(4), 103.
[11] Ding, Y., & Wu, H. (2019). Driver drowsiness detection with real time system
based on eye diameter detection and head angle position estimation. IEEE
Transactions on Intelligent Transportation Systems, 20(6), 2142-2152.
[12] Khan, M., & Lee, K. (2020).fatigue detection at real time by using (CNN)
model convolutional neural networks and framework. Journal of Intelligent
Transportation Systems, 24(1), 1-14.
[13] Alonso, J. A., & Suárez, J. M. (2021). A extensive survey based on real time
driver drowsiness detection. Journal of Transport and Health, 20, 101013.
[14] Li, Y., Zhang, C., & Zhao, J. (2020). Driver drowsiness detection using facial
expression and and deep neural network techniques. IEEE Access, 8, 189719-
189729.
[15] S. B. Bhatia, A. G. Vora, and R. D. Gohil, "Real-Time Driver Drowsiness
Detection System: A Review," International Journal of Computer Applications,
vol. 182, no. 24, pp. 1-7, 2019. DOI: 10.5120/ijca2019918377
[16] Dewi, C., Chen, R.C., Chang, C.W., Wu, S.H., Jiang, X., & Yu, H. (2022).
Eye Aspect Ratio for Real-Time Drowsiness Detection to Improve Driver Safety.
Electronics, 11(19), 3183.
[17] Roberts, J., & Adams, S." Exploring crossbred styles for Monitoring
automobilist Alertness Combining Biometric Sensors with Machine Learning
ways." (ITS )Intelligent Transport System's, vol -25, no -4, page 208- 222, 2024.
