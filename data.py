training_data = [
    ["Python is widely used for data science and AI applications.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    ["Many developers prefer JavaScript for frontend web development.", {'entities': [[21, 31, 'PROGRAMMING_LANGUAGE']]}],
    ["Java and C++ are commonly taught in computer science curricula.", {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE'], [9, 12, 'PROGRAMMING_LANGUAGE']]}],
    ["Ruby on Rails revolutionized web development workflows.", {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    ["Swift was developed by Apple for iOS and macOS development.", {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    ["PHP powers a significant percentage of websites on the internet.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    ["Rust offers memory safety without garbage collection.", {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    ["Go was created at Google to address performance concerns in large systems.", {'entities': [[0, 2, 'PROGRAMMING_LANGUAGE']]}],
    ["Kotlin has become the preferred language for Android development.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    ["TypeScript adds static typing to JavaScript.", {'entities': [[0, 10, 'PROGRAMMING_LANGUAGE'], [30, 40, 'PROGRAMMING_LANGUAGE']]}],
    ["COBOL is still used in many legacy banking systems.", {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    ["Scala combines object-oriented and functional programming paradigms.", {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    ["Haskell is known for its strong static typing and pure functional approach.", {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    ["R is primarily used for statistical computing and graphics.", {'entities': [[0, 1, 'PROGRAMMING_LANGUAGE']]}],
    ["Perl was once the dominant language for text processing and system administration.", {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    ["Developers using C# often work with the .NET framework.", {'entities': [[16, 18, 'PROGRAMMING_LANGUAGE']]}],
    ["Julia was designed for high-performance numerical analysis and computational science.", {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    ["Assembly language provides direct hardware access not available in high-level languages.", {'entities': [[0, 17, 'PROGRAMMING_LANGUAGE']]}],
    ["Objective-C was the primary language for iOS development before Swift.", {'entities': [[0, 11, 'PROGRAMMING_LANGUAGE'], [58, 63, 'PROGRAMMING_LANGUAGE']]}],
    ["Lua is commonly embedded in game engines for scripting.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    ["Fortran remains relevant in scientific computing despite its age.", {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    ["MATLAB excels at matrix operations and numerical computing.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    ["Clojure is a modern Lisp dialect running on the Java Virtual Machine.", {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE'], [22, 26, 'PROGRAMMING_LANGUAGE'], [47, 51, 'PROGRAMMING_LANGUAGE']]}],
    ["Groovy provides scripting capabilities for the Java platform.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE'], [45, 49, 'PROGRAMMING_LANGUAGE']]}],
    ["Erlang was designed for building massively scalable soft real-time systems.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    ["Prolog is primarily used for logic programming and AI applications.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    ["Dart is the language behind Flutter's cross-platform UI framework.", {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    ["Ada is still used in safety-critical systems like aviation and defense.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    ["Visual Basic .NET is integrated within Microsoft's development ecosystem.", {'entities': [[0, 18, 'PROGRAMMING_LANGUAGE']]}],
    ["F# combines functional programming with .NET infrastructure.", {'entities': [[0, 2, 'PROGRAMMING_LANGUAGE']]}],
    ["COBOL and Fortran are among the oldest programming languages still in use.", {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE'], [10, 17, 'PROGRAMMING_LANGUAGE']]}],
    ["Racket evolved from Scheme to become a language-building platform.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE'], [20, 26, 'PROGRAMMING_LANGUAGE']]}],
    ["Elixir builds on Erlang's VM for scalable, fault-tolerant applications.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE'], [16, 22, 'PROGRAMMING_LANGUAGE']]}],
    ["OCaml provides a blend of functional, imperative, and object-oriented paradigms.", {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    ["Smalltalk pioneered many object-oriented programming concepts.", {'entities': [[0, 9, 'PROGRAMMING_LANGUAGE']]}],
    ["PowerShell is Microsoft's task automation and configuration management framework.", {'entities': [[0, 10, 'PROGRAMMING_LANGUAGE']]}],
    ["Bash scripting is essential for Linux system administration.", {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    ["Crystal aims to have Ruby's elegance with C's performance.", {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE'], [22, 26, 'PROGRAMMING_LANGUAGE'], [39, 40, 'PROGRAMMING_LANGUAGE']]}],
    ["Nim combines Python's readability with C's efficiency.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE'], [13, 19, 'PROGRAMMING_LANGUAGE'], [39, 40, 'PROGRAMMING_LANGUAGE']]}],
    ["Zig provides an alternative to C with a focus on robustness and clarity.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE'], [28, 29, 'PROGRAMMING_LANGUAGE']]}],
    ["SQL is the standard language for database queries and manipulation.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    ["VBA enables automation in Microsoft Office applications.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    ["Pascal was designed to encourage good programming practices.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    ["BASIC was created to make programming accessible to beginners.", {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    ["PL/SQL extends SQL with procedural programming features for Oracle databases.", {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE'], [16, 19, 'PROGRAMMING_LANGUAGE']]}],
    ["D aims to improve upon C++ while maintaining compatibility.", {'entities': [[0, 1, 'PROGRAMMING_LANGUAGE'], [23, 26, 'PROGRAMMING_LANGUAGE']]}],
    ["Python and JavaScript consistently rank among the most popular programming languages.", {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE'], [11, 21, 'PROGRAMMING_LANGUAGE']]}],
    ["Developers often use C for embedded systems programming.", {'entities': [[20, 21, 'PROGRAMMING_LANGUAGE']]}],
    ["PHP and JavaScript are fundamental web development languages.", {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE'], [8, 18, 'PROGRAMMING_LANGUAGE']]}],
    ["Both Java and Python offer extensive libraries for various applications.", {'entities': [[5, 9, 'PROGRAMMING_LANGUAGE'], [14, 20, 'PROGRAMMING_LANGUAGE']]}],
    ["React has transformed how developers build user interfaces.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    ["Angular provides a comprehensive solution for single-page applications.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    ["TensorFlow and PyTorch are the leading machine learning frameworks.", {'entities': [[0, 10, 'FRAMEWORK_LIBRARY'], [15, 22, 'FRAMEWORK_LIBRARY']]}],
    ["Django simplifies web development with Python.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Node.js enables JavaScript to run on the server side.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    ["Spring Boot streamlines Java application development.", {'entities': [[0, 11, 'FRAMEWORK_LIBRARY']]}],
    ["Vue.js has gained popularity for its simplicity and flexibility.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Pandas is the go-to library for data manipulation in Python.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Express is a minimal web framework for Node.js applications.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [37, 44, 'FRAMEWORK_LIBRARY']]}],
    ["NumPy forms the foundation of scientific computing in Python.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    ["Selenium automates browser testing for web applications.", {'entities': [[0, 8, 'FRAMEWORK_LIBRARY']]}],
    ["Flask offers a lightweight alternative to Django for Python web development.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY'], [40, 46, 'FRAMEWORK_LIBRARY']]}],
    ["jQuery dominated front-end development before modern frameworks appeared.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Requests simplifies HTTP operations in Python applications.", {'entities': [[0, 8, 'FRAMEWORK_LIBRARY']]}],
    ["Scikit-learn provides machine learning tools for Python developers.", {'entities': [[0, 12, 'FRAMEWORK_LIBRARY']]}],
    ["Bootstrap makes responsive web design accessible to developers.", {'entities': [[0, 9, 'FRAMEWORK_LIBRARY']]}],
    ["Matplotlib creates publication-quality visualizations in Python.", {'entities': [[0, 10, 'FRAMEWORK_LIBRARY']]}],
    ["Laravel is a popular PHP framework with elegant syntax.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    ["Unity powers games across multiple platforms and devices.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    ["Keras provides a high-level API for TensorFlow and other neural network libraries.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY'], [34, 44, 'FRAMEWORK_LIBRARY']]}],
    ["Ruby on Rails pioneered convention over configuration in web frameworks.", {'entities': [[0, 13, 'FRAMEWORK_LIBRARY']]}],
    ["Apache Kafka handles real-time data streaming at scale.", {'entities': [[0, 12, 'FRAMEWORK_LIBRARY']]}],
    ["Symfony components are used in many PHP projects including Laravel.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [51, 58, 'FRAMEWORK_LIBRARY']]}],
    ["Redux manages state in complex React applications.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY'], [32, 37, 'FRAMEWORK_LIBRARY']]}],
    ["Lodash simplifies JavaScript data manipulation operations.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Hadoop processes large datasets across distributed computing clusters.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Jest is a popular testing framework for JavaScript applications.", {'entities': [[0, 4, 'FRAMEWORK_LIBRARY']]}],
    ["Unreal Engine powers many high-fidelity video games and simulations.", {'entities': [[0, 13, 'FRAMEWORK_LIBRARY']]}],
    ["Electron enables building desktop applications with web technologies.", {'entities': [[0, 8, 'FRAMEWORK_LIBRARY']]}],
    ["Tailwind CSS has changed how developers approach styling web applications.", {'entities': [[0, 12, 'FRAMEWORK_LIBRARY']]}],
    ["ASP.NET Core provides cross-platform web application development for .NET.", {'entities': [[0, 11, 'FRAMEWORK_LIBRARY']]}],
    ["Hibernate simplifies database operations in Java applications.", {'entities': [[0, 9, 'FRAMEWORK_LIBRARY']]}],
    ["Svelte compiles components to highly efficient JavaScript at build time.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["OpenCV is essential for computer vision applications.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Xamarin allows developers to build native mobile apps with C#.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    ["FastAPI creates high-performance Python APIs with minimal code.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    ["Pytest offers advanced testing capabilities for Python projects.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    ["Spark processes data at scale for big data applications.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    ["Docker Compose simplifies multi-container Docker application management.", {'entities': [[0, 13, 'FRAMEWORK_LIBRARY'], [43, 49, 'FRAMEWORK_LIBRARY']]}],
    ["Gatsby generates static websites with React and GraphQL.", {'entities': [[0, 6, 'FRAMEWORK_LIBRARY'], [36, 41, 'FRAMEWORK_LIBRARY']]}],
    ["Next.js provides server-side rendering capabilities for React applications.", {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [56, 61, 'FRAMEWORK_LIBRARY']]}],
    ["Spring Security handles authentication and authorization in Java applications.", {'entities': [[0, 15, 'FRAMEWORK_LIBRARY']]}],
    ["Kubernetes orchestrates containerized applications at scale.", {'entities': [[0, 10, 'FRAMEWORK_LIBRARY']]}],
    ["JUnit is the standard testing framework for Java applications.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    ["Mocha is a flexible JavaScript test framework running on Node.js.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY'], [51, 58, 'FRAMEWORK_LIBRARY']]}],
    ["Numpy and Pandas form the backbone of data analysis in Python.", {'entities': [[0, 5, 'FRAMEWORK_LIBRARY'], [10, 16, 'FRAMEWORK_LIBRARY']]}],
    ["Both React and Angular compete for frontend development mindshare.", {'entities': [[5, 10, 'FRAMEWORK_LIBRARY'], [15, 22, 'FRAMEWORK_LIBRARY']]}],
    ["Developers using Spring Boot often incorporate Hibernate for database operations.", {'entities': [[16, 27, 'FRAMEWORK_LIBRARY'], [45, 54, 'FRAMEWORK_LIBRARY']]}],
    ["NLTK provides tools for natural language processing in Python applications.", {'entities': [[0, 4, 'FRAMEWORK_LIBRARY']]}],
    ["The new Nvidia RTX 4090 offers unprecedented gaming performance.", {'entities': [[8, 22, 'HARDWARE']]}],
    ["Intel Core i9 processors target high-performance computing applications.", {'entities': [[0, 13, 'HARDWARE']]}],
    ["Raspberry Pi is popular for DIY electronics projects.", {'entities': [[0, 12, 'HARDWARE']]}],
    ["AMD Ryzen CPUs offer excellent performance for the price.", {'entities': [[0, 9, 'HARDWARE']]}],
    ["SSD storage provides faster data access than traditional hard drives.", {'entities': [[0, 3, 'HARDWARE']]}],
    ["Apple M1 chips deliver impressive performance with low power consumption.", {'entities': [[0, 8, 'HARDWARE']]}],
    ["Arduino boards are commonly used in introductory electronics courses.", {'entities': [[0, 7, 'HARDWARE']]}],
    ["The latest Samsung 990 PRO SSD achieves sequential read speeds of 7,450 MB/s.", {'entities': [[11, 25, 'HARDWARE']]}],
    ["Mechanical keyboards are preferred by many programmers for their tactile feedback.", {'entities': [[0, 20, 'HARDWARE']]}],
    ["HDMI cables connect displays to computers and media devices.", {'entities': [[0, 11, 'HARDWARE']]}],
    ["Logitech MX Master mouse features ergonomic design and customizable buttons.", {'entities': [[0, 18, 'HARDWARE']]}],
    ["DRAM is volatile memory that requires constant power to maintain data.", {'entities': [[0, 4, 'HARDWARE']]}],
    ["Dell XPS laptops balance performance and portability for professionals.", {'entities': [[0, 8, 'HARDWARE']]}],
    ["TPM modules provide hardware-based security functions.", {'entities': [[0, 3, 'HARDWARE']]}],
    ["Nvidia A100 GPUs accelerate machine learning workloads in data centers.", {'entities': [[0, 11, 'HARDWARE']]}],
    ["Western Digital hard drives offer high capacity storage solutions.", {'entities': [[0, 15, 'HARDWARE']]}],
    ["USB-C ports are becoming the standard connector for modern devices.", {'entities': [[0, 5, 'HARDWARE']]}],
    ["AirPods Pro deliver active noise cancellation in a compact design.", {'entities': [[0, 11, 'HARDWARE']]}],
    ["Motherboards connect all components of a computer system.", {'entities': [[0, 12, 'HARDWARE']]}],
    ["Qualcomm Snapdragon processors power many Android smartphones.", {'entities': [[0, 19, 'HARDWARE']]}],
    ["Kingston RAM modules provide reliable memory expansion options.", {'entities': [[0, 13, 'HARDWARE']]}],
    ["OLED displays offer superior contrast and color reproduction.", {'entities': [[0, 11, 'HARDWARE']]}],
    ["ASUS ROG gaming laptops target enthusiast gamers with high-end components.", {'entities': [[0, 8, 'HARDWARE']]}],
    ["CPU coolers prevent processors from overheating during intensive tasks.", {'entities': [[0, 10, 'HARDWARE']]}],
    ["Seagate external hard drives provide portable backup solutions.", {'entities': [[0, 23, 'HARDWARE']]}],
    ["Power supply units convert AC to DC power for computer components.", {'entities': [[0, 17, 'HARDWARE']]}],
    ["Graphics cards handle rendering tasks to offload the CPU.", {'entities': [[0, 14, 'HARDWARE'], [46, 49, 'HARDWARE']]}],
    ["FPGA chips can be reprogrammed for different processing tasks.", {'entities': [[0, 4, 'HARDWARE']]}],
    ["Sound cards process audio signals for high-quality sound output.", {'entities': [[0, 11, 'HARDWARE']]}],
    ["Network interface cards connect computers to networks.", {'entities': [[0, 24, 'HARDWARE']]}],
    ["Webcams became essential hardware during the remote work boom.", {'entities': [[0, 7, 'HARDWARE']]}],
    ["The Samsung 980 PRO and WD Black SN850 compete in the high-end NVMe market.", {'entities': [[4, 19, 'HARDWARE'], [24, 36, 'HARDWARE'], [61, 65, 'HARDWARE']]}],
    ["Mesh routers provide whole-home WiFi coverage without dead spots.", {'entities': [[0, 12, 'HARDWARE'], [30, 34, 'HARDWARE']]}],
    ["Intel Xeon processors are designed for server applications.", {'entities': [[0, 10, 'HARDWARE']]}],
    ["Surge protectors safeguard electronic devices from power spikes.", {'entities': [[0, 15, 'HARDWARE']]}],
    ["LG UltraFine monitors are popular among creative professionals.", {'entities': [[0, 13, 'HARDWARE']]}],
    ["Bluetooth speakers enable wireless audio streaming from mobile devices.", {'entities': [[0, 18, 'HARDWARE']]}],
    ["Solid state drives have largely replaced HDDs in modern laptops.", {'entities': [[0, 19, 'HARDWARE'], [38, 42, 'HARDWARE']]}],
    ["Thunderbolt ports offer high-speed data transfer and display output.", {'entities': [[0, 15, 'HARDWARE']]}],
    ["The IBM Model M keyboard remains popular decades after its introduction.", {'entities': [[4, 17, 'HARDWARE']]}],
    ["RAID arrays combine multiple drives for improved performance or redundancy.", {'entities': [[0, 11, 'HARDWARE']]}],
    ["Both AMD Radeon and Nvidia GeForce compete in the consumer GPU market.", {'entities': [[5, 15, 'HARDWARE'], [20, 33, 'HARDWARE'], [58, 61, 'HARDWARE']]}],
    ["Ethernet cables connect devices to local area networks.", {'entities': [[0, 15, 'HARDWARE']]}],
    ["Smart watches monitor health metrics and deliver notifications.", {'entities': [[0, 12, 'HARDWARE']]}],
    ["CPU and GPU temperatures should be monitored during intensive tasks.", {'entities': [[0, 3, 'HARDWARE'], [8, 11, 'HARDWARE']]}],
    ["The Raspberry Pi 5 and Arduino Mega offer different capabilities for maker projects.", {'entities': [[4, 18, 'HARDWARE'], [23, 35, 'HARDWARE']]}],
    ["PCIe slots allow expansion cards to connect to the motherboard.", {'entities': [[0, 9, 'HARDWARE'], [25, 40, 'HARDWARE'], [58, 69, 'HARDWARE']]}],
    ["The Apple MacBook Pro features the latest M3 chipset.", {'entities': [[4, 20, 'HARDWARE'], [39, 41, 'HARDWARE']]}],
    ["RAM and SSD upgrades can significantly improve system performance.", {'entities': [[0, 3, 'HARDWARE'], [8, 11, 'HARDWARE']]}],
    ["K-means clustering groups data points based on feature similarity.", {'entities': [[0, 16, 'ALGORITHM_MODEL']]}],
    ["BERT revolutionized natural language processing with bidirectional context.", {'entities': [[0, 4, 'ALGORITHM_MODEL']]}],
    ["Random Forest combines multiple decision trees to improve prediction accuracy.", {'entities': [[0, 13, 'ALGORITHM_MODEL'], [32, 45, 'ALGORITHM_MODEL']]}],
    ["GPT-4 demonstrates remarkable natural language generation capabilities.", {'entities': [[0, 5, 'ALGORITHM_MODEL']]}],
    ["Linear regression finds the relationship between independent and dependent variables.", {'entities': [[0, 16, 'ALGORITHM_MODEL']]}],
    ["The PageRank algorithm was fundamental to Google's early search engine success.", {'entities': [[4, 20, 'ALGORITHM_MODEL']]}],
    ["ResNet introduced residual connections to train deeper neural networks.", {'entities': [[0, 6, 'ALGORITHM_MODEL']]}],
    ["Dijkstra's algorithm finds the shortest path in a weighted graph.", {'entities': [[0, 19, 'ALGORITHM_MODEL']]}],
    ["Naive Bayes classifiers are based on applying Bayes' theorem with independence assumptions.", {'entities': [[0, 11, 'ALGORITHM_MODEL'], [42, 56, 'ALGORITHM_MODEL']]}],
    ["Transformer models have largely replaced RNNs for sequence modeling tasks.", {'entities': [[0, 11, 'ALGORITHM_MODEL'], [38, 42, 'ALGORITHM_MODEL']]}],
    ["YOLO enables real-time object detection in computer vision applications.", {'entities': [[0, 4, 'ALGORITHM_MODEL']]}],
    ["A* search algorithm efficiently finds paths in graph traversal problems.", {'entities': [[0, 17, 'ALGORITHM_MODEL']]}],
    ["LSTM networks are designed to handle the vanishing gradient problem in sequence data.", {'entities': [[0, 4, 'ALGORITHM_MODEL']]}],
    ["XGBoost implements gradient boosting with regularization features.", {'entities': [[0, 7, 'ALGORITHM_MODEL'], [19, 36, 'ALGORITHM_MODEL']]}],
    ["Convolutional Neural Networks excel at image processing tasks.", {'entities': [[0, 29, 'ALGORITHM_MODEL']]}],
    ["The SHA-256 algorithm generates fixed-size hash values.", {'entities': [[4, 11, 'ALGORITHM_MODEL']]}],
    ["Decision trees split data based on feature values to make predictions.", {'entities': [[0, 14, 'ALGORITHM_MODEL']]}],
    ["LLaMA and Llama 2 are open-source large language models.", {'entities': [[0, 5, 'ALGORITHM_MODEL'], [10, 17, 'ALGORITHM_MODEL']]}],
    ["Support Vector Machines find optimal hyperplanes to separate data classes.", {'entities': [[0, 23, 'ALGORITHM_MODEL']]}],
    ["Q-learning is a model-free reinforcement learning algorithm.", {'entities': [[0, 10, 'ALGORITHM_MODEL'], [25, 48, 'ALGORITHM_MODEL']]}],
    ["RSA encryption relies on the computational difficulty of factoring large prime numbers.", {'entities': [[0, 3, 'ALGORITHM_MODEL']]}],
    ["AlphaGo demonstrated superhuman performance in the game of Go.", {'entities': [[0, 7, 'ALGORITHM_MODEL']]}],
    ["Principal Component Analysis reduces data dimensionality while preserving variance.", {'entities': [[0, 28, 'ALGORITHM_MODEL']]}],
    ["T5 and BART are encoder-decoder transformer models for text generation.", {'entities': [[0, 2, 'ALGORITHM_MODEL'], [7, 11, 'ALGORITHM_MODEL'], [27, 37, 'ALGORITHM_MODEL']]}],
    ["K-nearest neighbors classification is intuitive but computationally expensive.", {'entities': [[0, 20, 'ALGORITHM_MODEL']]}],
    ["Binary search requires sorted data to efficiently locate values.", {'entities': [[0, 13, 'ALGORITHM_MODEL']]}],
    ["GANs consist of generator and discriminator networks competing against each other.", {'entities': [[0, 4, 'ALGORITHM_MODEL']]}],
    ["Quicksort typically outperforms other sorting algorithms for random data.", {'entities': [[0, 9, 'ALGORITHM_MODEL']]}],
    ["Word2Vec maps words to vector representations capturing semantic relationships.", {'entities': [[0, 8, 'ALGORITHM_MODEL']]}],
    ["DBSCAN clusters data based on density without requiring a predefined number of clusters.", {'entities': [[0, 6, 'ALGORITHM_MODEL']]}],
    ["Markov chains model sequences where each state depends only on the previous state.", {'entities': [[0, 13, 'ALGORITHM_MODEL']]}],
    ["ARIMA models forecast time series data with seasonal components.", {'entities': [[0, 5, 'ALGORITHM_MODEL']]}],
    ["The Fast Fourier Transform efficiently converts signals between time and frequency domains.", {'entities': [[4, 25, 'ALGORITHM_MODEL']]}],
    ["Logistic regression predicts binary outcomes despite its name suggesting regression.", {'entities': [[0, 18, 'ALGORITHM_MODEL']]}],
    ["MobileNet optimizes convolutional neural networks for mobile devices.", {'entities': [[0, 9, 'ALGORITHM_MODEL'], [20, 49, 'ALGORITHM_MODEL']]}],
    ["Bubble sort is simple to implement but inefficient for large datasets.", {'entities': [[0, 11, 'ALGORITHM_MODEL']]}],
    ["U-Net architecture excels at image segmentation tasks.", {'entities': [[0, 5, 'ALGORITHM_MODEL']]}],
    ["The AES encryption standard provides strong security for sensitive data.", {'entities': [[4, 7, 'ALGORITHM_MODEL']]}],
    ["Gradient descent iteratively minimizes error functions in machine learning models.", {'entities': [[0, 16, 'ALGORITHM_MODEL']]}],
    ["The minimax algorithm is fundamental to strategic decision-making in game theory.", {'entities': [[4, 19, 'ALGORITHM_MODEL']]}],
    ["ViT applies transformer architecture to image recognition tasks.", {'entities': [[0, 3, 'ALGORITHM_MODEL'], [12, 22, 'ALGORITHM_MODEL']]}],
    ["Hierarchical clustering organizes data into a tree-like structure of nested clusters.", {'entities': [[0, 22, 'ALGORITHM_MODEL']]}],
    ["MD5 hashing is no longer considered secure for cryptographic applications.", {'entities': [[0, 3, 'ALGORITHM_MODEL']]}],
    ["Deep Q-Networks combine Q-learning with deep neural networks.", {'entities': [[0, 16, 'ALGORITHM_MODEL'], [25, 35, 'ALGORITHM_MODEL'], [41, 62, 'ALGORITHM_MODEL']]}],
    ["The Apriori algorithm identifies frequent itemsets in transaction databases.", {'entities': [[4, 20, 'ALGORITHM_MODEL']]}],
    ["Both BERT and RoBERTa have advanced the state of natural language understanding.", {'entities': [[5, 9, 'ALGORITHM_MODEL'], [14, 21, 'ALGORITHM_MODEL']]}],
    ["GPT-3 and DALL-E demonstrate different applications of transformer architectures.", {'entities': [[0, 5, 'ALGORITHM_MODEL'], [10, 16, 'ALGORITHM_MODEL'], [59, 69, 'ALGORITHM_MODEL']]}],
    ["CNN and RNN architectures serve different purposes in deep learning.", {'entities': [[0, 3, 'ALGORITHM_MODEL'], [8, 11, 'ALGORITHM_MODEL'], [55, 68, 'ALGORITHM_MODEL']]}],
    ["The combination of Random Forest and Gradient Boosting often yields robust predictions.", {'entities': [[17, 30, 'ALGORITHM_MODEL'], [35, 51, 'ALGORITHM_MODEL']]}],
    ["HTTP forms the foundation of data communication on the web.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["HTTPS adds encryption to HTTP for secure communication.", {'entities': [[0, 5, 'PROTOCOL'], [24, 28, 'PROTOCOL']]}],
    ["TCP ensures reliable, ordered data delivery between applications.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["UDP prioritizes speed over reliability for real-time applications.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["SSH provides secure remote access to systems and servers.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["FTP is commonly used for transferring files between clients and servers.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["SMTP handles email transmission between servers.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["DNS translates domain names to IP addresses for network routing.", {'entities': [[0, 3, 'PROTOCOL'], [34, 36, 'PROTOCOL']]}],
    ["DHCP automatically assigns IP addresses to devices on a network.", {'entities': [[0, 4, 'PROTOCOL'], [26, 28, 'PROTOCOL']]}],
    ["IMAP allows email clients to access messages stored on mail servers.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["Bluetooth enables wireless communication between nearby devices.", {'entities': [[0, 9, 'PROTOCOL']]}],
    ["POP3 downloads email from servers to local clients.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["LDAP provides directory services for user authentication.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["MQTT is designed for IoT devices with limited bandwidth and processing power.", {'entities': [[0, 4, 'PROTOCOL'], [19, 22, 'PROTOCOL']]}],
    ["WebSockets enable real-time bidirectional communication in web applications.", {'entities': [[0, 11, 'PROTOCOL']]}],
    ["NTP synchronizes clocks across computer networks.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["SIP is a signaling protocol for initiating and maintaining real-time sessions.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["SNMP monitors and manages network devices and their functions.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["TLS provides privacy and data integrity between communicating applications.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["RTP transports audio and video over IP networks for streaming applications.", {'entities': [[0, 3, 'PROTOCOL'], [35, 37, 'PROTOCOL']]}],
    ["RTSP controls streaming media servers in real-time applications.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["SMB allows file and printer sharing between networked computers.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["BGP routes traffic between autonomous systems on the internet.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["OAuth provides secure delegated access to server resources.", {'entities': [[0, 5, 'PROTOCOL']]}],
    ["IPv6 expands the address space available for internet-connected devices.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["ARP maps IP addresses to MAC addresses on local networks.", {'entities': [[0, 3, 'PROTOCOL'], [9, 11, 'PROTOCOL'], [22, 25, 'PROTOCOL']]}],
    ["XMPP enables real-time communication and presence information.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["ICMP sends error messages and operational information about network conditions.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["BitTorrent distributes file sharing across peer-to-peer networks.", {'entities': [[0, 10, 'PROTOCOL']]}],
    ["IPsec secures IP communication by authenticating and encrypting packets.", {'entities': [[0, 5, 'PROTOCOL'], [14, 16, 'PROTOCOL']]}],
    ["gRPC enables efficient communication between microservices.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["WebRTC enables real-time communication directly in web browsers.", {'entities': [[0, 6, 'PROTOCOL']]}],
    ["Telnet provides text-based remote login functionality.", {'entities': [[0, 6, 'PROTOCOL']]}],
    ["OSPF routes packets within a single autonomous system.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["QUIC improves performance of connection-oriented web applications.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["WPA3 secures wireless networks with stronger encryption methods.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["CoAP is designed for internet of things devices with limited resources.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["SFTP adds encryption to FTP for secure file transfers.", {'entities': [[0, 4, 'PROTOCOL'], [21, 24, 'PROTOCOL']]}],
    ["REST defines a set of constraints for creating web services.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["SOAP exchanges structured information in web services implementation.", {'entities': [[0, 4, 'PROTOCOL']]}],
    ["RDP enables remote connections to another computer over a network.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["NetBIOS provides services for local network communications.", {'entities': [[0, 7, 'PROTOCOL']]}],
    ["ZigBee is designed for low-power wireless personal area networks.", {'entities': [[0, 6, 'PROTOCOL']]}],
    ["IGMP manages multicast group memberships on IP networks.", {'entities': [[0, 4, 'PROTOCOL'], [43, 45, 'PROTOCOL']]}],
    ["Z-Wave enables wireless communication for home automation devices.", {'entities': [[0, 6, 'PROTOCOL']]}],
    ["LwM2M standardizes communication between IoT devices and servers.", {'entities': [[0, 5, 'PROTOCOL'], [33, 36, 'PROTOCOL']]}],
    ["Modbus connects industrial electronic devices in automation systems.", {'entities': [[0, 6, 'PROTOCOL']]}],
    ["Both HTTP and HTTPS protocols are fundamental to web communication.", {'entities': [[5, 9, 'PROTOCOL'], [14, 19, 'PROTOCOL']]}],
    ["SSH and SFTP provide secure alternatives to Telnet and FTP respectively.", {'entities': [[0, 3, 'PROTOCOL'], [8, 12, 'PROTOCOL'], [45, 51, 'PROTOCOL'], [56, 59, 'PROTOCOL']]}],
    ["The combination of TCP and IP forms the foundation of internet communication.", {'entities': [[17, 20, 'PROTOCOL'], [25, 27, 'PROTOCOL']]}],
    ["SSL/TLS protocols encrypt data transmitted between web clients and servers.", {'entities': [[0, 7, 'PROTOCOL']]}],
    ["NFC enables short-range wireless communication between compatible devices.", {'entities': [[0, 3, 'PROTOCOL']]}],
    ["LoRaWAN is designed for wide-area networks with low power requirements.", {'entities': [[0, 7, 'PROTOCOL']]}],
    ["IMAP and POP3 are email retrieval protocols with different capabilities.", {'entities': [[0, 4, 'PROTOCOL'], [9, 13, 'PROTOCOL']]}],
    ["DNS and DHCP are critical network services for modern internet connectivity.", {'entities': [[0, 3, 'PROTOCOL'], [8, 12, 'PROTOCOL']]}],
    ["SMTP, IMAP, and POP3 handle different aspects of email communication.", {'entities': [[0, 4, 'PROTOCOL'], [6, 10, 'PROTOCOL'], [16, 20, 'PROTOCOL']]}],
    ["The WebSocket protocol enables full-duplex communication channels over TCP.", {'entities': [[4, 13, 'PROTOCOL'], [66, 69, 'PROTOCOL']]}],
    ["Both UDP and TCP operate at the transport layer of the OSI model.", {'entities': [[5, 8, 'PROTOCOL'], [13, 16, 'PROTOCOL'], [54, 57, 'PROTOCOL']]}],
    ["MQTT and CoAP are lightweight protocols designed for IoT applications.", {'entities': [[0, 4, 'PROTOCOL'], [9, 13, 'PROTOCOL'], [35, 38, 'PROTOCOL']]}],
    ["HTTP/3 uses QUIC instead of TCP for improved performance.", {'entities': [[0, 6, 'PROTOCOL'], [12, 16, 'PROTOCOL'], [28, 31, 'PROTOCOL']]}],
    ["PDF documents preserve formatting across different platforms.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["JPEG compression balances image quality with file size.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["JSON is widely used for data interchange between web services.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["MP3 files revolutionized digital music distribution.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["HTML documents form the structure of web pages.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["CSV files store tabular data in plain text format.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["XML provides a flexible way to define structured data.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["PNG supports lossless compression and transparency.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["DOCX is the default format for Microsoft Word documents.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["MP4 containers can store video, audio, and subtitle tracks.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["XLSX spreadsheets organize data in rows and columns.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["GIF images support animation and were popular in early web design.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["SVG graphics scale without losing quality at any resolution.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["TIFF images are commonly used in publishing and professional photography.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["WAV files provide uncompressed audio with high fidelity.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["YAML offers a human-readable data serialization standard.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["ZIP archives compress multiple files into a single container.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["EXE files are executable programs on Windows systems.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["RAR provides efficient compression for file archives.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["PPTX presentations include slides, notes, and multimedia elements.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["WEBP images offer better compression than JPEG with similar quality.", {'entities': [[0, 4, 'FILE_FORMAT'], [44, 48, 'FILE_FORMAT']]}],
    ["OBJ files store 3D geometry data for models and scenes.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["EPUB is the standard format for digital books and publications.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["MOV files are Apple's QuickTime video container format.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["SQL scripts define database schemas and operations.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["FLAC provides lossless audio compression for archival purposes.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["PSD files preserve Photoshop editing capabilities and layers.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["AVI is an older video container format developed by Microsoft.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["OGG is an open container format for multimedia content.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["TXT files contain plain text without formatting information.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["ICO files contain icons for Windows applications and websites.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["BMP images store bitmap data without compression.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["AVIF is a newer image format offering efficient compression.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["MKV containers support multiple video, audio, and subtitle tracks.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["TTF and OTF are font file formats with different capabilities.", {'entities': [[0, 3, 'FILE_FORMAT'], [8, 11, 'FILE_FORMAT']]}],
    ["DWG files contain proprietary AutoCAD drawing data.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["APK packages contain Android application files.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["RTF documents support basic text formatting across platforms.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["SRT files contain subtitle timing information for videos.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["STL files describe surface geometry for 3D printing.", {'entities': [[0, 3, 'FILE_FORMAT']]}],
    ["HDF5 provides a hierarchical data format for scientific datasets.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["HEIC images offer high efficiency compression on Apple devices.", {'entities': [[0, 4, 'FILE_FORMAT']]}],
    ["PDF, DOCX, and XLSX are common document formats in business environments.", {'entities': [[0, 3, 'FILE_FORMAT'], [5, 9, 'FILE_FORMAT'], [15, 19, 'FILE_FORMAT']]}],
    ["JPEG, PNG, and GIF are widely used image formats on the web.", {'entities': [[0, 4, 'FILE_FORMAT'], [6, 9, 'FILE_FORMAT'], [15, 18, 'FILE_FORMAT']]}],
    ["MP3, WAV, and FLAC represent different approaches to audio compression.", {'entities': [[0, 3, 'FILE_FORMAT'], [5, 8, 'FILE_FORMAT'], [14, 18, 'FILE_FORMAT']]}],
    ["CSV and JSON are popular formats for data exchange between systems.", {'entities': [[0, 3, 'FILE_FORMAT'], [8, 12, 'FILE_FORMAT']]}],
    ["Both XML and YAML provide structured data serialization options.", {'entities': [[5, 8, 'FILE_FORMAT'], [13, 17, 'FILE_FORMAT']]}],
    ["MP4 and MKV containers store video content with different feature sets.", {'entities': [[0, 3, 'FILE_FORMAT'], [8, 11, 'FILE_FORMAT']]}],
    ["The choice between PNG and JPEG depends on the image content and use case.", {'entities': [[17, 20, 'FILE_FORMAT'], [25, 29, 'FILE_FORMAT']]}],
    ["A firewall protects networks by filtering incoming and outgoing traffic.", {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    ["Malware includes viruses, trojans, and other malicious software.", {'entities': [[0, 7, 'CYBERSECURITY_TERM'], [17, 24, 'CYBERSECURITY_TERM'], [26, 33, 'CYBERSECURITY_TERM']]}],
    ["Zero-day vulnerabilities are unknown to software vendors and lack patches.", {'entities': [[0, 8, 'CYBERSECURITY_TERM'], [9, 23, 'CYBERSECURITY_TERM']]}],
    ["Phishing attacks trick users into revealing sensitive information.", {'entities': [[0, 8, 'CYBERSECURITY_TERM'], [9, 16, 'CYBERSECURITY_TERM']]}],
    ["Encryption converts data into unreadable code to prevent unauthorized access.", {'entities': [[0, 10, 'CYBERSECURITY_TERM']]}],
    ["A VPN creates a secure connection over public networks.", {'entities': [[2, 5, 'CYBERSECURITY_TERM']]}],
    ["Two-factor authentication adds an extra layer of security beyond passwords.", {'entities': [[0, 27, 'CYBERSECURITY_TERM']]}],
    ["Ransomware encrypts files and demands payment for decryption keys.", {'entities': [[0, 10, 'CYBERSECURITY_TERM'], [54, 70, 'CYBERSECURITY_TERM']]}],
    ["SQL injection attacks exploit vulnerable database queries.", {'entities': [[0, 13, 'CYBERSECURITY_TERM'], [14, 21, 'CYBERSECURITY_TERM']]}],
    ["DDoS attacks overwhelm servers with excessive traffic.", {'entities': [[0, 4, 'CYBERSECURITY_TERM'], [5, 12, 'CYBERSECURITY_TERM']]}],
    ["A security token provides temporary authentication credentials.", {'entities': [[2, 16, 'CYBERSECURITY_TERM'], [36, 50, 'CYBERSECURITY_TERM']]}],
    ["Penetration testing evaluates system security by simulating attacks.", {'entities': [[0, 19, 'CYBERSECURITY_TERM']]}],
    ["A honeypot attracts attackers to monitor their tactics.", {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    ["Social engineering manipulates people into divulging confidential information.", {'entities': [[0, 18, 'CYBERSECURITY_TERM']]}],
    ["An intrusion detection system monitors networks for suspicious activity.", {'entities': [[3, 30, 'CYBERSECURITY_TERM']]}],
    ["Biometric authentication uses physical characteristics for identity verification.", {'entities': [[0, 25, 'CYBERSECURITY_TERM'], [66, 88, 'CYBERSECURITY_TERM']]}],
    ["A man-in-the-middle attack intercepts communications between two parties.", {'entities': [[2, 24, 'CYBERSECURITY_TERM'], [25, 31, 'CYBERSECURITY_TERM']]}],
    ["The principle of least privilege restricts access rights to the minimum necessary.", {'entities': [[4, 32, 'CYBERSECURITY_TERM']]}],
    ["A rootkit enables unauthorized access to a computer while hiding its existence.", {'entities': [[2, 9, 'CYBERSECURITY_TERM'], [27, 33, 'CYBERSECURITY_TERM']]}],
    ["Threat modeling identifies and prioritizes potential security risks.", {'entities': [[0, 15, 'CYBERSECURITY_TERM'], [45, 60, 'CYBERSECURITY_TERM']]}],
    ["Exploit kits automate cyberattacks using known vulnerabilities.", {'entities': [[0, 12, 'CYBERSECURITY_TERM'], [22, 34, 'CYBERSECURITY_TERM'], [41, 55, 'CYBERSECURITY_TERM']]}],
    ["Hash functions generate fixed-size outputs from variable-size inputs.", {'entities': [[0, 14, 'CYBERSECURITY_TERM']]}],
    ["Spyware collects information without the user's knowledge or consent.", {'entities': [[0, 7, 'CYBERSECURITY_TERM']]}],
    ["Access control lists define permissions for system resources.", {'entities': [[0, 20, 'CYBERSECURITY_TERM'], [28, 39, 'CYBERSECURITY_TERM']]}],
    ["A sandbox isolates programs to prevent malicious code from spreading.", {'entities': [[2, 9, 'CYBERSECURITY_TERM'], [41, 55, 'CYBERSECURITY_TERM']]}],
    ["A backdoor provides unauthorized access to a system while bypassing security measures.", {'entities': [[2, 10, 'CYBERSECURITY_TERM'], [29, 35, 'CYBERSECURITY_TERM']]}],
    ["Cryptojacking hijacks computing resources to mine cryptocurrency.", {'entities': [[0, 13, 'CYBERSECURITY_TERM']]}],
    ["A cross-site scripting attack injects malicious code into trusted websites.", {'entities': [[2, 27, 'CYBERSECURITY_TERM'], [28, 34, 'CYBERSECURITY_TERM'], [42, 56, 'CYBERSECURITY_TERM']]}],
    ["Buffer overflow exploits occur when programs write data beyond allocated memory.", {'entities': [[0, 15, 'CYBERSECURITY_TERM'], [16, 24, 'CYBERSECURITY_TERM']]}],
    ["Digital signatures verify the authenticity and integrity of messages.", {'entities': [[0, 18, 'CYBERSECURITY_TERM'], [33, 45, 'CYBERSECURITY_TERM'], [50, 59, 'CYBERSECURITY_TERM']]}],
    ["Keyloggers record keystrokes to capture sensitive information.", {'entities': [[0, 10, 'CYBERSECURITY_TERM']]}],
    ["A botnet consists of compromised computers controlled by attackers.", {'entities': [[2, 8, 'CYBERSECURITY_TERM'], [22, 33, 'CYBERSECURITY_TERM']]}],
    ["The CIA triad defines confidentiality, integrity, and availability goals.", {'entities': [[4, 13, 'CYBERSECURITY_TERM'], [22, 37, 'CYBERSECURITY_TERM'], [39, 48, 'CYBERSECURITY_TERM'], [54, 66, 'CYBERSECURITY_TERM']]}],
    ["Security through obscurity relies on secrecy rather than robust design.", {'entities': [[0, 28, 'CYBERSECURITY_TERM']]}],
    ["CSRF attacks trick users into performing unwanted actions on authenticated websites.", {'entities': [[0, 4, 'CYBERSECURITY_TERM'], [5, 12, 'CYBERSECURITY_TERM'], [54, 67, 'CYBERSECURITY_TERM']]}],
    ["A zero-trust model assumes no implicit trust in any user or system.", {'entities': [[2, 18, 'CYBERSECURITY_TERM'], [34, 39, 'CYBERSECURITY_TERM']]}],
    ["The cyber kill chain describes stages of cyberattacks for defensive planning.", {'entities': [[4, 19, 'CYBERSECURITY_TERM'], [39, 51, 'CYBERSECURITY_TERM']]}],
    ["Vulnerability scanning identifies potential security weaknesses in systems.", {'entities': [[0, 23, 'CYBERSECURITY_TERM'], [42, 61, 'CYBERSECURITY_TERM']]}],
    ["A denial-of-service attack makes resources unavailable to intended users.", {'entities': [[2, 25, 'CYBERSECURITY_TERM'], [26, 32, 'CYBERSECURITY_TERM']]}],
    ["Both phishing and spear phishing attempt to steal sensitive information, but target different scopes.", {'entities': [[5, 13, 'CYBERSECURITY_TERM'], [18, 32, 'CYBERSECURITY_TERM'], [50, 72, 'CYBERSECURITY_TERM']]}],
    ["Encryption and hashing protect data in different ways with distinct security purposes.", {'entities': [[0, 10, 'CYBERSECURITY_TERM'], [15, 22, 'CYBERSECURITY_TERM'], [40, 49, 'CYBERSECURITY_TERM'], [61, 69, 'CYBERSECURITY_TERM']]}],
    ["A combination of firewalls, IDS, and antivirus software creates defense in depth.", {'entities': [[16, 25, 'CYBERSECURITY_TERM'], [27, 30, 'CYBERSECURITY_TERM'], [36, 45, 'CYBERSECURITY_TERM'], [66, 81, 'CYBERSECURITY_TERM']]}],
    ["Threat actors include hackers, nation-states, and insider threats.", {'entities': [[0, 13, 'CYBERSECURITY_TERM'], [22, 29, 'CYBERSECURITY_TERM'], [31, 44, 'CYBERSECURITY_TERM'], [50, 65, 'CYBERSECURITY_TERM']]}],
    ["Both authentication and authorization are critical for access control systems.", {'entities': [[5, 19, 'CYBERSECURITY_TERM'], [24, 36, 'CYBERSECURITY_TERM'], [54, 68, 'CYBERSECURITY_TERM']]}],
    ["The security team conducted penetration testing and vulnerability scanning to identify weaknesses.", {'entities': [[25, 44, 'CYBERSECURITY_TERM'], [49, 72, 'CYBERSECURITY_TERM']]}],
    ["A secure SDLC integrates security throughout the software development lifecycle.", {'entities': [[9, 13, 'CYBERSECURITY_TERM'], [25, 33, 'CYBERSECURITY_TERM'], [48, 74, 'CYBERSECURITY_TERM']]}],
    ["Implementing MFA reduces the risk of credential theft and account takeovers.", {'entities': [[13, 16, 'CYBERSECURITY_TERM'], [38, 55, 'CYBERSECURITY_TERM'], [60, 78, 'CYBERSECURITY_TERM']]}],
    ["Security incidents require proper incident response and forensic analysis.", {'entities': [[0, 19, 'CYBERSECURITY_TERM'], [37, 54, 'CYBERSECURITY_TERM'], [59, 75, 'CYBERSECURITY_TERM']]}],

    ['React is a JavaScript library for building user interfaces.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],

    ['Angular is a platform for building mobile and desktop apps.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Vue.js is a progressive JavaScript framework.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Django is a high-level Python web framework.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Flask is a lightweight Python web framework.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Spring is a framework for building Java applications.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Laravel is a PHP framework for web artisans.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Express.js is a Node.js web application framework.', {'entities': [[0, 10, 'FRAMEWORK_LIBRARY']]}],
    
    ['Ruby on Rails is a server-side web application framework.', {'entities': [[0, 13, 'FRAMEWORK_LIBRARY']]}],
    
    ['TensorFlow is an open-source machine learning library.', {'entities': [[0, 10, 'FRAMEWORK_LIBRARY']]}],
    
    ['PyTorch is a machine learning library developed by Facebook.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Keras is a high-level neural networks API.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Bootstrap is a front-end framework for responsive design.', {'entities': [[0, 9, 'FRAMEWORK_LIBRARY']]}],
    
    ['jQuery is a fast and concise JavaScript library.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Ember.js is a framework for ambitious web applications.', {'entities': [[0, 8, 'FRAMEWORK_LIBRARY']]}],
    
    ['Svelte is a modern JavaScript framework.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Next.js is a React framework for server-side rendering.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Nuxt.js is a Vue.js framework for universal applications.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Flutter is a UI toolkit for building natively compiled apps.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Xamarin is a framework for building cross-platform apps.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Cordova is a platform for building mobile apps with web tech.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Ionic is a framework for building hybrid mobile apps.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['ASP.NET is a framework for building web applications.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Symfony is a PHP framework for web projects.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['CakePHP is a rapid development framework for PHP.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['CodeIgniter is a lightweight PHP framework.', {'entities': [[0, 11, 'FRAMEWORK_LIBRARY']]}],
    
    ['Zend Framework is a PHP framework for enterprise apps.', {'entities': [[0, 14, 'FRAMEWORK_LIBRARY']]}],
    
    ['Hibernate is a Java framework for object-relational mapping.', {'entities': [[0, 9, 'FRAMEWORK_LIBRARY']]}],
    
    ['Struts is a framework for building Java web applications.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Play Framework is a reactive web framework for Java.', {'entities': [[0, 14, 'FRAMEWORK_LIBRARY']]}],
    
    ['Grails is a Groovy-based framework for web applications.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Meteor is a full-stack JavaScript framework.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Backbone.js is a lightweight JavaScript framework.', {'entities': [[0, 11, 'FRAMEWORK_LIBRARY']]}],
    
    ['Aurelia is a modern JavaScript framework.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Polymer is a library for building web components.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Three.js is a JavaScript library for 3D graphics.', {'entities': [[0, 8, 'FRAMEWORK_LIBRARY']]}],
    
    ['D3.js is a JavaScript library for data visualization.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Chart.js is a simple JavaScript charting library.', {'entities': [[0, 8, 'FRAMEWORK_LIBRARY']]}],
    
    ['Socket.IO is a library for real-time web applications.', {'entities': [[0, 9, 'FRAMEWORK_LIBRARY']]}],
    
    ['Axios is a promise-based HTTP client for the browser.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Lodash is a utility library for JavaScript.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Moment.js is a library for parsing and formatting dates.', {'entities': [[0, 9, 'FRAMEWORK_LIBRARY']]}],
    
    ['Underscore.js is a JavaScript utility library.', {'entities': [[0, 13, 'FRAMEWORK_LIBRARY']]}],
    
    ['Redux is a state management library for JavaScript apps.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['MobX is a state management library for React apps.', {'entities': [[0, 4, 'FRAMEWORK_LIBRARY']]}],
    
    ['GraphQL is a query language for APIs.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Apollo is a GraphQL client for React applications.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Jest is a JavaScript testing framework.', {'entities': [[0, 4, 'FRAMEWORK_LIBRARY']]}],
    
    ['Mocha is a JavaScript test framework for Node.js.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Chai is an assertion library for Node.js.', {'entities': [[0, 4, 'FRAMEWORK_LIBRARY']]}],
    
    ['Python is a popular programming language.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Java is widely used for enterprise applications.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['JavaScript is essential for web development.', {'entities': [[0, 10, 'PROGRAMMING_LANGUAGE']]}],
    
    ['C++ is known for its performance and efficiency.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Ruby is often used for building web applications.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Go is a statically typed language developed by Google.', {'entities': [[0, 2, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Swift is the primary language for iOS development.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Kotlin is fully interoperable with Java.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Rust is gaining popularity for system-level programming.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['TypeScript adds static typing to JavaScript.', {'entities': [[0, 10, 'PROGRAMMING_LANGUAGE']]}],
    
    ['PHP is commonly used for server-side scripting.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Perl is known for its text processing capabilities.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Scala combines object-oriented and functional programming.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Dart is used for building mobile and web apps.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Haskell is a purely functional programming language.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Lua is often embedded in applications for scripting.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Elixir is built on the Erlang virtual machine.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Clojure is a dialect of Lisp for the JVM.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['F# is a functional-first language for .NET.', {'entities': [[0, 2, 'PROGRAMMING_LANGUAGE']]}],
    
    ['R is widely used for statistical computing.', {'entities': [[0, 1, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Objective-C was the main language for macOS development.', {'entities': [[0, 11, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Bash is a shell scripting language for Unix systems.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['PowerShell is a task automation framework.', {'entities': [[0, 11, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Groovy is a dynamic language for the Java platform.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Julia is designed for high-performance numerical analysis.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Ada is used in safety-critical systems.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['COBOL is still used in legacy systems.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Fortran is used in scientific computing.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Prolog is a logic programming language.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Erlang is known for its concurrency support.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['OCaml is a functional language with type inference.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Smalltalk is an object-oriented programming language.', {'entities': [[0, 9, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Scheme is a minimalist dialect of Lisp.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Racket is a general-purpose programming language.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['VHDL is used for hardware description.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Verilog is another hardware description language.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['ABAP is used for SAP application development.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['D is a systems programming language.', {'entities': [[0, 1, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Nim is a statically typed compiled language.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Zig is a general-purpose programming language.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Crystal is inspired by Ruby and focuses on performance.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Elm is a functional language for front-end development.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['ReasonML is a syntax extension for OCaml.', {'entities': [[0, 8, 'PROGRAMMING_LANGUAGE']]}],
    
    ['PureScript is a strongly-typed functional language.', {'entities': [[0, 10, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Idris is a general-purpose functional language.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['ATS is a language with formal verification features.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Agda is a dependently typed functional language.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Coq is used for formal verification of software.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Isabelle is a proof assistant and programming language.', {'entities': [[0, 9, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Lean is a functional programming language.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['The CPU is the brain of the computer.', {'entities': [[4, 7, 'HARDWARE']]}],
    
    ['GPUs are essential for rendering graphics.', {'entities': [[0, 4, 'HARDWARE']]}],
    
    ['SSDs are faster than traditional hard drives.', {'entities': [[0, 4, 'HARDWARE']]}],
    
    ['The motherboard connects all the components.', {'entities': [[4, 15, 'HARDWARE']]}],
    
    ['RAM is used for temporary data storage.', {'entities': [[0, 3, 'HARDWARE']]}],
    
    ['The power supply unit provides electricity to the system.', {'entities': [[4, 20, 'HARDWARE']]}],
    
    ['A cooling fan prevents the system from overheating.', {'entities': [[2, 13, 'HARDWARE']]}],
    
    ['The keyboard and mouse are input devices.', {'entities': [[4, 12, 'HARDWARE'], [17, 22, 'HARDWARE']]}],
    
    ['The monitor displays the output from the computer.', {'entities': [[4, 11, 'HARDWARE']]}],
    
    ['A printer is used to produce physical copies of documents.', {'entities': [[2, 9, 'HARDWARE']]}],
    
    ['The hard drive stores all the data permanently.', {'entities': [[4, 14, 'HARDWARE']]}],
    
    ['The graphics card enhances visual performance.', {'entities': [[4, 18, 'HARDWARE']]}],
    
    ['The sound card improves audio quality.', {'entities': [[4, 14, 'HARDWARE']]}],
    
    ['A network adapter connects the computer to the internet.', {'entities': [[2, 17, 'HARDWARE']]}],
    
    ['The optical drive reads CDs and DVDs.', {'entities': [[4, 17, 'HARDWARE']]}],
    
    ['A USB flash drive is portable and convenient.', {'entities': [[2, 16, 'HARDWARE']]}],
    
    ['The case houses all the internal components.', {'entities': [[4, 8, 'HARDWARE']]}],
    
    ['The heatsink dissipates heat from the CPU.', {'entities': [[4, 12, 'HARDWARE']]}],
    
    ['A microphone is used for voice input.', {'entities': [[2, 12, 'HARDWARE']]}],
    
    ['The scanner converts physical documents into digital files.', {'entities': [[4, 11, 'HARDWARE']]}],
    
    ['A webcam is used for video conferencing.', {'entities': [[2, 8, 'HARDWARE']]}],
    
    ['The router connects multiple devices to the internet.', {'entities': [[4, 10, 'HARDWARE']]}],
    
    ['A modem converts digital signals to analog signals.', {'entities': [[2, 7, 'HARDWARE']]}],
    
    ['The touchpad is an alternative to a mouse.', {'entities': [[4, 12, 'HARDWARE']]}],
    
    ['A joystick is used for gaming and simulations.', {'entities': [[2, 10, 'HARDWARE']]}],
    
    ['The projector displays images on a large screen.', {'entities': [[4, 13, 'HARDWARE']]}],
    
    ['A barcode scanner reads product information.', {'entities': [[2, 18, 'HARDWARE']]}],
    
    ['The server rack organizes multiple servers.', {'entities': [[4, 15, 'HARDWARE']]}],
    
    ['A docking station connects laptops to peripherals.', {'entities': [[2, 16, 'HARDWARE']]}],
    
    ['The stylus is used for drawing on touchscreens.', {'entities': [[4, 10, 'HARDWARE']]}],
    
    ['A VR headset provides an immersive experience.', {'entities': [[2, 12, 'HARDWARE']]}],
    
    ['The smartwatch tracks fitness and health metrics.', {'entities': [[4, 13, 'HARDWARE']]}],
    
    ['A drone is used for aerial photography.', {'entities': [[2, 7, 'HARDWARE']]}],
    
    ['The e-reader is designed for reading digital books.', {'entities': [[4, 12, 'HARDWARE']]}],
    
    ['A 3D printer creates physical objects from digital models.', {'entities': [[2, 12, 'HARDWARE']]}],
    
    ['The smart thermostat regulates home temperature.', {'entities': [[4, 19, 'HARDWARE']]}],
    
    ['A fitness tracker monitors physical activity.', {'entities': [[2, 16, 'HARDWARE']]}],
    
    ['The gaming console provides entertainment.', {'entities': [[4, 18, 'HARDWARE']]}],
    
    ['A smart speaker responds to voice commands.', {'entities': [[2, 14, 'HARDWARE']]}],
    
    ['The external hard drive provides additional storage.', {'entities': [[4, 21, 'HARDWARE']]}],
    
    ['A graphics tablet is used for digital art.', {'entities': [[2, 17, 'HARDWARE']]}],
    
    ['The NAS device stores data for network access.', {'entities': [[4, 13, 'HARDWARE']]}],
    
    ['A biometric scanner verifies identity.', {'entities': [[2, 18, 'HARDWARE']]}],
    
    ['The smart lock secures doors with digital keys.', {'entities': [[4, 13, 'HARDWARE']]}],
    
    ['A robotic vacuum cleans floors automatically.', {'entities': [[2, 17, 'HARDWARE']]}],
    
    ['The smart bulb can be controlled via a smartphone.', {'entities': [[4, 13, 'HARDWARE']]}],
    
    ['A home assistant device manages smart home systems.', {'entities': [[2, 22, 'HARDWARE']]}],
    
    ['The gaming headset provides immersive audio.', {'entities': [[4, 18, 'HARDWARE']]}],
    
    ['A thermal printer produces receipts and labels.', {'entities': [[2, 17, 'HARDWARE']]}],
    
    ['The point-of-sale system processes transactions.', {'entities': [[4, 22, 'HARDWARE']]}],
    
    ['The k-means algorithm is used for clustering data.', {'entities': [[4, 11, 'ALGORITHM_MODEL']]}],
    
    ['Linear regression is a statistical modeling technique.', {'entities': [[0, 16, 'ALGORITHM_MODEL']]}],
    
    ['Decision trees are used for classification tasks.', {'entities': [[0, 14, 'ALGORITHM_MODEL']]}],
    
    ['The random forest algorithm improves prediction accuracy.', {'entities': [[4, 19, 'ALGORITHM_MODEL']]}],
    
    ['Support vector machines are powerful for classification.', {'entities': [[0, 22, 'ALGORITHM_MODEL']]}],
    
    ['Neural networks are inspired by the human brain.', {'entities': [[0, 15, 'ALGORITHM_MODEL']]}],
    
    ['The gradient boosting algorithm is used in machine learning.', {'entities': [[4, 22, 'ALGORITHM_MODEL']]}],
    
    ['K-nearest neighbors is a simple classification algorithm.', {'entities': [[0, 19, 'ALGORITHM_MODEL']]}],
    
    ['Principal component analysis reduces dimensionality.', {'entities': [[0, 26, 'ALGORITHM_MODEL']]}],
    
    ['The Apriori algorithm is used for association rule mining.', {'entities': [[4, 18, 'ALGORITHM_MODEL']]}],
    
    ['Naive Bayes is a probabilistic classification model.', {'entities': [[0, 11, 'ALGORITHM_MODEL']]}],
    
    ['The PageRank algorithm ranks web pages by importance.', {'entities': [[4, 14, 'ALGORITHM_MODEL']]}],
    
    ['Logistic regression is used for binary classification.', {'entities': [[0, 19, 'ALGORITHM_MODEL']]}],
    
    ['The AdaBoost algorithm improves weak classifiers.', {'entities': [[4, 13, 'ALGORITHM_MODEL']]}],
    
    ['Hidden Markov models are used for sequence prediction.', {'entities': [[0, 20, 'ALGORITHM_MODEL']]}],
    
    ['The DBSCAN algorithm is used for density-based clustering.', {'entities': [[4, 12, 'ALGORITHM_MODEL']]}],
    
    ['XGBoost is a popular gradient boosting framework.', {'entities': [[0, 7, 'ALGORITHM_MODEL']]}],
    
    ['The EM algorithm is used for parameter estimation.', {'entities': [[4, 14, 'ALGORITHM_MODEL']]}],
    
    ['The Perceptron is a simple neural network model.', {'entities': [[4, 14, 'ALGORITHM_MODEL']]}],
    
    ['The t-SNE algorithm is used for data visualization.', {'entities': [[4, 10, 'ALGORITHM_MODEL']]}],
    
    ['The Q-learning algorithm is used in reinforcement learning.', {'entities': [[4, 17, 'ALGORITHM_MODEL']]}],
    
    ['The Viterbi algorithm is used for decoding sequences.', {'entities': [[4, 18, 'ALGORITHM_MODEL']]}],
    
    ['The LSTM model is used for sequential data processing.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The GAN model generates realistic data samples.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The CNN model is widely used in image processing.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The RNN model is used for time series analysis.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The BERT model is used for natural language processing.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The GPT model generates human-like text.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The YOLO algorithm is used for object detection.', {'entities': [[4, 9, 'ALGORITHM_MODEL']]}],
    
    ['The SVM model is used for classification and regression.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The KNN algorithm is simple and effective.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The PCA algorithm reduces the number of variables.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The ARIMA model is used for time series forecasting.', {'entities': [[4, 10, 'ALGORITHM_MODEL']]}],
    
    ['The Markov chain model predicts future states.', {'entities': [[4, 17, 'ALGORITHM_MODEL']]}],
    
    ['The Monte Carlo algorithm is used for simulations.', {'entities': [[4, 18, 'ALGORITHM_MODEL']]}],
    
    ['The Dijkstra algorithm finds the shortest path.', {'entities': [[4, 17, 'ALGORITHM_MODEL']]}],
    
    ['The Floyd-Warshall algorithm solves all-pairs shortest paths.', {'entities': [[4, 22, 'ALGORITHM_MODEL']]}],
    
    ['The Bellman-Ford algorithm handles negative weights.', {'entities': [[4, 19, 'ALGORITHM_MODEL']]}],
    
    ['The Kruskal algorithm is used for finding minimum spanning trees.', {'entities': [[4, 15, 'ALGORITHM_MODEL']]}],
    
    ['The Prim algorithm is another approach for minimum spanning trees.', {'entities': [[4, 10, 'ALGORITHM_MODEL']]}],
    
    ['The A* algorithm is used for pathfinding in games.', {'entities': [[4, 9, 'ALGORITHM_MODEL']]}],
    
    ['The RSA algorithm is used for encryption.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The AES algorithm is a symmetric encryption standard.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The DES algorithm is an older encryption method.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The SHA algorithm is used for hashing data.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The MD5 algorithm is used for checksums.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The ElGamal algorithm is used for public-key cryptography.', {'entities': [[4, 15, 'ALGORITHM_MODEL']]}],
    
    ['The Diffie-Hellman algorithm enables secure key exchange.', {'entities': [[4, 20, 'ALGORITHM_MODEL']]}],
    
    ['The ECC algorithm is used for elliptic curve cryptography.', {'entities': [[4, 8, 'ALGORITHM_MODEL']]}],
    
    ['The Simplex algorithm is used for linear programming.', {'entities': [[4, 15, 'ALGORITHM_MODEL']]}],
    
    ['HTTP is the foundation of data communication on the web.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['HTTPS ensures secure communication over the internet.', {'entities': [[0, 5, 'PROTOCOL']]}],
    
    ['FTP is used for transferring files between systems.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['SMTP is the protocol for sending emails.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['POP3 is used for retrieving emails from a server.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['IMAP allows users to manage emails on a server.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['TCP ensures reliable data delivery.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['UDP is faster but less reliable than TCP.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['DNS translates domain names into IP addresses.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['DHCP assigns IP addresses to devices on a network.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['SSH provides secure remote access to systems.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['Telnet is an older protocol for remote access.', {'entities': [[0, 6, 'PROTOCOL']]}],
    
    ['RDP is used for remote desktop connections.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['SNMP is used for managing network devices.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['NTP synchronizes clocks across networks.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['ICMP is used for error reporting in networks.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['ARP resolves IP addresses to MAC addresses.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['RTP is used for delivering audio and video over the internet.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['RTSP controls streaming media servers.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['SIP is used for initiating communication sessions.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['LDAP is used for accessing directory services.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['BGP is a protocol for routing between autonomous systems.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['OSPF is a routing protocol for IP networks.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['EIGRP is a Cisco proprietary routing protocol.', {'entities': [[0, 5, 'PROTOCOL']]}],
    
    ['RIP is a simple routing protocol.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['IPsec provides secure communication over IP networks.', {'entities': [[0, 5, 'PROTOCOL']]}],
    
    ['SSL ensures secure communication between clients and servers.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['TLS is the successor to SSL.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['SFTP provides secure file transfers.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['TFTP is a simpler version of FTP.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['MQTT is a lightweight messaging protocol.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['CoAP is a protocol for constrained devices.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['WebSocket enables real-time communication.', {'entities': [[0, 10, 'PROTOCOL']]}],
    
    ['HTTP/2 improves web performance.', {'entities': [[0, 6, 'PROTOCOL']]}],
    
    ['QUIC is a modern transport protocol.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['SMB is used for file sharing in Windows networks.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['NFS allows file sharing between Unix systems.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['AFP is used for file sharing in macOS.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['NetBIOS is an older protocol for network communication.', {'entities': [[0, 7, 'PROTOCOL']]}],
    
    ['PPTP is a VPN protocol.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['L2TP is used for VPN connections.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['OpenVPN is an open-source VPN protocol.', {'entities': [[0, 8, 'PROTOCOL']]}],
    
    ['IKE is used for key exchange in VPNs.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['GRE is a tunneling protocol.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['STP prevents loops in network topologies.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['VLAN is used for segmenting networks.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['CDP is a Cisco discovery protocol.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['LLDP is a vendor-neutral discovery protocol.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['RADIUS is used for network authentication.', {'entities': [[0, 6, 'PROTOCOL']]}],
    
    ['TACACS+ is a Cisco authentication protocol.', {'entities': [[0, 8, 'PROTOCOL']]}],
    
    ['PDF is a popular format for documents.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['JPEG is commonly used for images.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['PNG supports lossless compression.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['MP3 is a widely used audio format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['MP4 is a common video format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['DOCX is the default format for Word documents.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['XLSX is used for Excel spreadsheets.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['PPTX is the format for PowerPoint presentations.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['TXT is a simple text file format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['CSV is used for storing tabular data.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['JSON is a lightweight data interchange format.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['XML is used for structured data storage.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['HTML is the standard format for web pages.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['CSS is used for styling web pages.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['ZIP is a common compression format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['RAR is another compression format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['GIF is used for animated images.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['BMP is a bitmap image format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['TIFF is used for high-quality images.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['SVG is a vector image format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['WAV is a lossless audio format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['FLAC is a high-quality audio format.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['AVI is a video container format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['MKV is a versatile video format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['MOV is a video format developed by Apple.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['OGG is an open multimedia container format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['WEBM is a video format for the web.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['ISO is a disk image format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['DMG is a disk image format for macOS.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['EXE is an executable file format for Windows.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['APK is the file format for Android apps.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['IPA is the file format for iOS apps.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['PY is the file extension for Python scripts.', {'entities': [[0, 2, 'FILE_FORMAT']]}],
    
    ['JAR is a Java archive file format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['SQL is a file format for database scripts.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['LOG is a file format for storing logs.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['INI is a configuration file format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['YAML is a human-readable data format.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['TOML is a configuration file format.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['RTF is a text file format with formatting.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['ODT is an open document text format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['ODS is an open document spreadsheet format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['ODP is an open document presentation format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['EPUB is an e-book file format.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['MOBI is another e-book file format.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['PSD is the file format for Photoshop documents.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['AI is the file format for Adobe Illustrator.', {'entities': [[0, 2, 'FILE_FORMAT']]}],
    
    ['DWG is a CAD file format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['STL is a file format for 3D models.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['FBX is a 3D model exchange format.', {'entities': [[0, 3, 'FILE_FORMAT']]}],
    
    ['Phishing is a common cyber attack.', {'entities': [[0, 8, 'CYBERSECURITY_TERM']]}],
    
    ['Malware is software designed to harm systems.', {'entities': [[0, 6, 'CYBERSECURITY_TERM']]}],
    
    ['Ransomware encrypts files and demands payment.', {'entities': [[0, 11, 'CYBERSECURITY_TERM']]}],
    
    ['A firewall protects networks from unauthorized access.', {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    
    ['Encryption ensures data confidentiality.', {'entities': [[0, 10, 'CYBERSECURITY_TERM']]}],
    
    ['Two-factor authentication adds an extra layer of security.', {'entities': [[0, 24, 'CYBERSECURITY_TERM']]}],
    
    ['A VPN provides secure remote access.', {'entities': [[2, 5, 'CYBERSECURITY_TERM']]}],
    
    ['A zero-day exploit targets unknown vulnerabilities.', {'entities': [[2, 16, 'CYBERSECURITY_TERM']]}],
    
    ['Social engineering manipulates users into revealing information.', {'entities': [[0, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A DDoS attack overwhelms a server with traffic.', {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    
    ['A botnet is a network of infected devices.', {'entities': [[2, 8, 'CYBERSECURITY_TERM']]}],
    
    ['A keylogger records keystrokes.', {'entities': [[2, 11, 'CYBERSECURITY_TERM']]}],
    
    ['A trojan horse disguises itself as legitimate software.', {'entities': [[2, 15, 'CYBERSECURITY_TERM']]}],
    
    ['A worm spreads without user interaction.', {'entities': [[2, 6, 'CYBERSECURITY_TERM']]}],
    
    ['A rootkit provides unauthorized access to a system.', {'entities': [[2, 9, 'CYBERSECURITY_TERM']]}],
    
    ['Penetration testing identifies system vulnerabilities.', {'entities': [[0, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A honeypot lures attackers to study their methods.', {'entities': [[2, 9, 'CYBERSECURITY_TERM']]}],
    
    ['A security patch fixes vulnerabilities.', {'entities': [[2, 16, 'CYBERSECURITY_TERM']]}],
    
    ['A vulnerability scanner detects weaknesses.', {'entities': [[2, 23, 'CYBERSECURITY_TERM']]}],
    
    ['A brute force attack tries all possible passwords.', {'entities': [[2, 16, 'CYBERSECURITY_TERM']]}],
    
    ['A man-in-the-middle attack intercepts communication.', {'entities': [[2, 24, 'CYBERSECURITY_TERM']]}],
    
    ['A SQL injection attack exploits database vulnerabilities.', {'entities': [[2, 17, 'CYBERSECURITY_TERM']]}],
    
    ['A cross-site scripting attack targets web applications.', {'entities': [[2, 23, 'CYBERSECURITY_TERM']]}],
    
    ['A data breach exposes sensitive information.', {'entities': [[2, 13, 'CYBERSECURITY_TERM']]}],
    
    ['A security audit evaluates system defenses.', {'entities': [[2, 16, 'CYBERSECURITY_TERM']]}],
    
    ['A security policy defines rules for protecting data.', {'entities': [[2, 17, 'CYBERSECURITY_TERM']]}],
    
    ['A security incident requires immediate response.', {'entities': [[2, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A security token provides authentication.', {'entities': [[2, 16, 'CYBERSECURITY_TERM']]}],
    
    ['A security certificate ensures secure communication.', {'entities': [[2, 21, 'CYBERSECURITY_TERM']]}],
    
    ['A security protocol defines secure communication rules.', {'entities': [[2, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A security framework provides guidelines for protection.', {'entities': [[2, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A security control mitigates risks.', {'entities': [[2, 17, 'CYBERSECURITY_TERM']]}],
    
    ['A security risk assessment identifies threats.', {'entities': [[2, 25, 'CYBERSECURITY_TERM']]}],
    
    ['A security breach compromises data integrity.', {'entities': [[2, 17, 'CYBERSECURITY_TERM']]}],
    
    ['A security vulnerability exposes systems to attacks.', {'entities': [[2, 23, 'CYBERSECURITY_TERM']]}],
    
    ['A security threat is a potential danger.', {'entities': [[2, 17, 'CYBERSECURITY_TERM']]}],
    
    ['A security measure protects against attacks.', {'entities': [[2, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A security awareness program educates users.', {'entities': [[2, 23, 'CYBERSECURITY_TERM']]}],
    
    ['A security analyst monitors system activity.', {'entities': [[2, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A security engineer designs secure systems.', {'entities': [[2, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A security consultant provides expert advice.', {'entities': [[2, 20, 'CYBERSECURITY_TERM']]}],
    
    ['A security administrator manages system defenses.', {'entities': [[2, 23, 'CYBERSECURITY_TERM']]}],
    
    ['A security architect designs secure infrastructures.', {'entities': [[2, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A security operations center monitors threats.', {'entities': [[2, 26, 'CYBERSECURITY_TERM']]}],
    
    ['A security information and event management system collects logs.', {'entities': [[2, 50, 'CYBERSECURITY_TERM']]}],
    
    ['A security baseline defines minimum protection standards.', {'entities': [[2, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A security clearance grants access to sensitive data.', {'entities': [[2, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A security violation breaches policies.', {'entities': [[2, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A security posture reflects an organization’s defenses.', {'entities': [[2, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A security culture promotes awareness and responsibility.', {'entities': [[2, 17, 'CYBERSECURITY_TERM']]}],
    
    ['I often use Python for data analysis because it’s so versatile.', {'entities': [[12, 18, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Do you know if Java is still popular for enterprise applications?', {'entities': [[16, 20, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I’m learning JavaScript to build interactive websites.', {'entities': [[14, 24, 'PROGRAMMING_LANGUAGE']]}],
    
    ['C++ can be challenging, but it’s great for performance.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Have you tried Ruby? It’s really beginner-friendly.', {'entities': [[13, 17, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I heard Go is gaining popularity for backend development.', {'entities': [[9, 11, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Swift is the language I use for iOS app development.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Kotlin is becoming the preferred language for Android apps.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I don’t know much about Rust, but it seems interesting.', {'entities': [[21, 25, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Why do people say Haskell is good for functional programming?', {'entities': [[22, 29, 'PROGRAMMING_LANGUAGE']]}],    
    
    ['I’ve been using React for my frontend projects lately.', {'entities': [[17, 22, 'FRAMEWORK_LIBRARY']]}],
    
    ['Do you think Angular is better than Vue.js?', {'entities': [[13, 20, 'FRAMEWORK_LIBRARY'], [31, 36, 'FRAMEWORK_LIBRARY']]}],
    
    ['I need to learn Django for my new web development job.', {'entities': [[16, 22, 'FRAMEWORK_LIBRARY']]}],
    
    ['Flask is so lightweight compared to other frameworks.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Have you worked with TensorFlow for machine learning?', {'entities': [[18, 28, 'FRAMEWORK_LIBRARY']]}],
    
    ['I’m not sure if I should use Bootstrap or Tailwind CSS.', {'entities': [[22, 31, 'FRAMEWORK_LIBRARY'], [35, 46, 'FRAMEWORK_LIBRARY']]}],
    
    ['Laravel makes PHP development so much easier.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['I’ve heard great things about Spring for Java applications.', {'entities': [[27, 33, 'FRAMEWORK_LIBRARY']]}],
    
    ['Is Flask a good choice for small projects?', {'entities': [[7, 12, 'FRAMEWORK_LIBRARY']]}],
    
    ['I need to focus on learning Express.js for backend development.', {'entities': [[28, 38, 'FRAMEWORK_LIBRARY']]}],
    
    ['My CPU is overheating; I think I need a better cooler.', {'entities': [[3, 6, 'HARDWARE']]}],
    
    ['Do you know if upgrading my GPU will improve gaming performance?', {'entities': [[25, 28, 'HARDWARE']]}],
    
    ['I’m looking for an SSD to speed up my computer.', {'entities': [[18, 21, 'HARDWARE']]}],
    
    ['The motherboard in my PC is outdated.', {'entities': [[4, 15, 'HARDWARE']]}],
    
    ['How much RAM do I need for video editing?', {'entities': [[12, 15, 'HARDWARE']]}],
    
    ['I think my power supply unit is failing.', {'entities': [[11, 27, 'HARDWARE']]}],
    
    ['Do I need a cooling fan for my new build?', {'entities': [[13, 24, 'HARDWARE']]}],
    
    ['My keyboard stopped working, so I need a new one.', {'entities': [[3, 11, 'HARDWARE']]}],
    
    ['The monitor I bought has a great display.', {'entities': [[4, 11, 'HARDWARE']]}],
    
    ['I’m thinking of getting a printer for my home office.', {'entities': [[22, 29, 'HARDWARE']]}],
    
    ['I’m studying the k-means algorithm for my data science class.', {'entities': [[18, 25, 'ALGORITHM_MODEL']]}],
    
    ['Do you know how linear regression works?', {'entities': [[13, 29, 'ALGORITHM_MODEL']]}],
    
    ['I need to implement a decision tree for my project.', {'entities': [[24, 38, 'ALGORITHM_MODEL']]}],
    
    ['Random forest is one of my favorite algorithms.', {'entities': [[0, 15, 'ALGORITHM_MODEL']]}],
    
    ['Have you ever used support vector machines?', {'entities': [[16, 38, 'ALGORITHM_MODEL']]}],
    
    ['Neural networks are so fascinating to learn about.', {'entities': [[0, 15, 'ALGORITHM_MODEL']]}],
    
    ['I’m trying to understand how gradient boosting works.', {'entities': [[26, 44, 'ALGORITHM_MODEL']]}],
    
    ['K-nearest neighbors is a simple but effective algorithm.', {'entities': [[0, 19, 'ALGORITHM_MODEL']]}],
    
    ['Principal component analysis is great for dimensionality reduction.', {'entities': [[0, 26, 'ALGORITHM_MODEL']]}],
    
    ['I’ve heard Apriori is good for market basket analysis.', {'entities': [[11, 18, 'ALGORITHM_MODEL']]}],
    
    ['I need to configure HTTP for my web server.', {'entities': [[18, 22, 'PROTOCOL']]}],
    
    ['Is HTTPS more secure than HTTP?', {'entities': [[3, 8, 'PROTOCOL'], [25, 29, 'PROTOCOL']]}],
    
    ['I’m having trouble setting up FTP on my computer.', {'entities': [[24, 27, 'PROTOCOL']]}],
    
    ['Do you know how SMTP works for sending emails?', {'entities': [[13, 17, 'PROTOCOL']]}],
    
    ['I need to learn more about TCP for my networking class.', {'entities': [[23, 26, 'PROTOCOL']]}],
    
    ['DNS is essential for translating domain names.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['I’m not familiar with SSH; can you explain it?', {'entities': [[18, 21, 'PROTOCOL']]}],
    
    ['RDP is used for remote desktop connections, right?', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['I need to configure SNMP for network monitoring.', {'entities': [[18, 22, 'PROTOCOL']]}],
    
    ['I saved the document as a PDF for easy sharing.', {'entities': [[25, 28, 'FILE_FORMAT']]}],
    
    ['Do you know how to convert a file to MP3?', {'entities': [[30, 33, 'FILE_FORMAT']]}],
    
    ['I prefer MP4 over AVI for video files.', {'entities': [[10, 13, 'FILE_FORMAT'], [19, 22, 'FILE_FORMAT']]}],
    
    ['The report is in DOCX format; can you open it?', {'entities': [[18, 22, 'FILE_FORMAT']]}],
    
    ['I need to export this data to CSV for analysis.', {'entities': [[28, 31, 'FILE_FORMAT']]}],
    
    ['JSON is my preferred format for APIs.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['XML is so verbose compared to JSON.', {'entities': [[0, 3, 'FILE_FORMAT'], [24, 28, 'FILE_FORMAT']]}],
    
    ['I’m not sure how to open a ZIP file.', {'entities': [[20, 23, 'FILE_FORMAT']]}],
    
    ['Can you send me the file in TXT format?', {'entities': [[29, 32, 'FILE_FORMAT']]}],
    
    ['Phishing attacks are becoming more sophisticated.', {'entities': [[0, 8, 'CYBERSECURITY_TERM']]}],
    
    ['I need to install antivirus software to protect against malware.', {'entities': [[48, 54, 'CYBERSECURITY_TERM']]}],
    
    ['Ransomware locked all my files until I paid the fee.', {'entities': [[0, 11, 'CYBERSECURITY_TERM']]}],
    
    ['A firewall is essential for network security.', {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    
    ['Encryption is the best way to secure sensitive data.', {'entities': [[0, 10, 'CYBERSECURITY_TERM']]}],
    
    ['Two-factor authentication adds an extra layer of security.', {'entities': [[0, 24, 'CYBERSECURITY_TERM']]}],
    
    ['I use a VPN to protect my online privacy.', {'entities': [[9, 12, 'CYBERSECURITY_TERM']]}],
    
    ['Zero-day exploits are hard to defend against.', {'entities': [[0, 14, 'CYBERSECURITY_TERM']]}],
    
    ['Social engineering is a common tactic for hackers.', {'entities': [[0, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A DDoS attack can take down an entire website.', {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    
    ['I’ve been using Python for data analysis, and it’s amazing.', {'entities': [[17, 23, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Do you think Java is still relevant in 2023?', {'entities': [[13, 17, 'PROGRAMMING_LANGUAGE']]}],
    
    ['JavaScript is everywhere on the web.', {'entities': [[0, 10, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I’m struggling with C++; it’s so complex.', {'entities': [[18, 21, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Ruby is such an elegant language for scripting.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Go is perfect for building scalable backend systems.', {'entities': [[0, 2, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Swift makes iOS development so much easier.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Kotlin is now the preferred language for Android.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I’ve heard Rust is great for system-level programming.', {'entities': [[11, 15, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Haskell is so different from other languages.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I’m learning TypeScript to improve my JavaScript code.', {'entities': [[14, 24, 'PROGRAMMING_LANGUAGE']]}],
    
    ['PHP is still widely used for web development.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Perl is great for text processing tasks.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Scala combines the best of object-oriented and functional programming.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Dart is the language behind Flutter.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Lua is often used in game development.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Julia is gaining popularity in data science.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['R is my go-to language for statistical analysis.', {'entities': [[13, 14, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Objective-C is still used in some legacy iOS projects.', {'entities': [[0, 11, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Bash is essential for scripting on Unix systems.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I’m using React for my new project, and it’s fantastic.', {'entities': [[11, 16, 'FRAMEWORK_LIBRARY']]}],
    
    ['Angular is a bit complex, but it’s powerful.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Vue.js is so easy to learn compared to other frameworks.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Django is perfect for building web applications quickly.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Flask is lightweight and great for small projects.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Spring is the go-to framework for Java developers.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Laravel makes PHP development so much easier.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Express.js is my favorite for building APIs.', {'entities': [[0, 10, 'FRAMEWORK_LIBRARY']]}],
    
    ['Ruby on Rails is still popular for web development.', {'entities': [[0, 13, 'FRAMEWORK_LIBRARY']]}],
    
    ['TensorFlow is amazing for machine learning projects.', {'entities': [[0, 10, 'FRAMEWORK_LIBRARY']]}],
    
    ['PyTorch is gaining traction in the AI community.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Keras is so user-friendly for building neural networks.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY']]}],
    
    ['Bootstrap is my go-to for responsive design.', {'entities': [[0, 9, 'FRAMEWORK_LIBRARY']]}],
    
    ['jQuery is outdated, but it’s still used in some projects.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Ember.js is great for ambitious web applications.', {'entities': [[0, 8, 'FRAMEWORK_LIBRARY']]}],
    
    ['Svelte is a game-changer for frontend development.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY']]}],
    
    ['Next.js is perfect for server-side rendering.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Nuxt.js makes Vue.js development so much easier.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Flutter is amazing for building cross-platform apps.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['Xamarin is great for C# developers building mobile apps.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY']]}],
    
    ['My CPU is running at 100% all the time.', {'entities': [[3, 6, 'HARDWARE']]}],
    
    ['I need to upgrade my GPU for better gaming performance.', {'entities': [[19, 22, 'HARDWARE']]}],
    
    ['SSDs are so much faster than traditional hard drives.', {'entities': [[0, 4, 'HARDWARE']]}],
    
    ['The motherboard in my PC is outdated and needs replacement.', {'entities': [[4, 15, 'HARDWARE']]}],
    
    ['How much RAM do I need for multitasking?', {'entities': [[12, 15, 'HARDWARE']]}],
    
    ['My power supply unit is making weird noises.', {'entities': [[3, 19, 'HARDWARE']]}],
    
    ['I think my cooling fan is broken.', {'entities': [[11, 22, 'HARDWARE']]}],
    
    ['I spilled coffee on my keyboard, and now it’s not working.', {'entities': [[20, 28, 'HARDWARE']]}],
    
    ['The monitor I bought has a 4K resolution.', {'entities': [[4, 11, 'HARDWARE']]}],
    
    ['I need a new printer for my home office.', {'entities': [[12, 19, 'HARDWARE']]}],
    
    ['My hard drive is almost full; I need to clean it up.', {'entities': [[3, 13, 'HARDWARE']]}],
    
    ['The graphics card in my PC is outdated.', {'entities': [[4, 18, 'HARDWARE']]}],
    
    ['I’m looking for a sound card to improve audio quality.', {'entities': [[18, 28, 'HARDWARE']]}],
    
    ['Do I need a network adapter for better internet speed?', {'entities': [[13, 28, 'HARDWARE']]}],
    
    ['My optical drive stopped reading discs.', {'entities': [[3, 16, 'HARDWARE']]}],
    
    ['I lost my USB flash drive; it had important files.', {'entities': [[11, 25, 'HARDWARE']]}],
    
    ['The case for my PC is too small for all the components.', {'entities': [[4, 8, 'HARDWARE']]}],
    
    ['I need a new heatsink for my CPU.', {'entities': [[15, 23, 'HARDWARE']]}],
    
    ['My microphone is not picking up sound properly.', {'entities': [[3, 13, 'HARDWARE']]}],
    
    ['I’m thinking of buying a scanner for digitizing documents.', {'entities': [[23, 30, 'HARDWARE']]}],
    
    ['I’m trying to implement k-means for clustering my data.', {'entities': [[23, 30, 'ALGORITHM_MODEL']]}],
    
    ['Linear regression is the first algorithm I learned.', {'entities': [[0, 16, 'ALGORITHM_MODEL']]}],
    
    ['Decision trees are great for classification tasks.', {'entities': [[0, 14, 'ALGORITHM_MODEL']]}],
    
    ['Random forest is one of the most robust algorithms.', {'entities': [[0, 15, 'ALGORITHM_MODEL']]}],
    
    ['Support vector machines are powerful but complex.', {'entities': [[0, 22, 'ALGORITHM_MODEL']]}],
    
    ['Neural networks are inspired by the human brain.', {'entities': [[0, 15, 'ALGORITHM_MODEL']]}],
    
    ['Gradient boosting is used in many machine learning competitions.', {'entities': [[0, 18, 'ALGORITHM_MODEL']]}],
    
    ['K-nearest neighbors is simple but effective.', {'entities': [[0, 19, 'ALGORITHM_MODEL']]}],
    
    ['Principal component analysis helps reduce dimensionality.', {'entities': [[0, 26, 'ALGORITHM_MODEL']]}],
    
    ['Apriori is commonly used for market basket analysis.', {'entities': [[0, 7, 'ALGORITHM_MODEL']]}],
    
    ['Naive Bayes is a probabilistic classification model.', {'entities': [[0, 11, 'ALGORITHM_MODEL']]}],
    
    ['PageRank is the algorithm behind Google’s search engine.', {'entities': [[0, 9, 'ALGORITHM_MODEL']]}],
    
    ['Logistic regression is used for binary classification.', {'entities': [[0, 19, 'ALGORITHM_MODEL']]}],
    
    ['AdaBoost is an ensemble learning technique.', {'entities': [[0, 9, 'ALGORITHM_MODEL']]}],
    
    ['Hidden Markov models are used for sequence prediction.', {'entities': [[0, 20, 'ALGORITHM_MODEL']]}],
    
    ['DBSCAN is great for density-based clustering.', {'entities': [[0, 7, 'ALGORITHM_MODEL']]}],
    
    ['XGBoost is a popular gradient boosting framework.', {'entities': [[0, 7, 'ALGORITHM_MODEL']]}],
    
    ['The EM algorithm is used for parameter estimation.', {'entities': [[4, 15, 'ALGORITHM_MODEL']]}],
    
    ['The Perceptron is a simple neural network model.', {'entities': [[4, 14, 'ALGORITHM_MODEL']]}],
    
    ['t-SNE is used for visualizing high-dimensional data.', {'entities': [[0, 6, 'ALGORITHM_MODEL']]}],
    
    ['HTTP is the backbone of the web.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['HTTPS is more secure than HTTP.', {'entities': [[0, 5, 'PROTOCOL'], [24, 28, 'PROTOCOL']]}],
    
    ['FTP is used for transferring large files.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['SMTP is essential for sending emails.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['POP3 is used for retrieving emails from a server.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['IMAP allows you to manage emails on a server.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['TCP ensures reliable data delivery over networks.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['UDP is faster but less reliable than TCP.', {'entities': [[0, 3, 'PROTOCOL'], [30, 33, 'PROTOCOL']]}],
    
    ['DNS translates domain names into IP addresses.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['DHCP automatically assigns IP addresses to devices.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['SSH is used for secure remote access to servers.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['Telnet is an older protocol for remote access.', {'entities': [[0, 6, 'PROTOCOL']]}],
    
    ['RDP is great for remote desktop connections.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['SNMP is used for managing network devices.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['NTP synchronizes clocks across networks.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['ICMP is used for error reporting in networks.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['ARP resolves IP addresses to MAC addresses.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['RTP is used for delivering audio and video.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['RTSP controls streaming media servers.', {'entities': [[0, 4, 'PROTOCOL']]}],
    
    ['SIP is used for initiating communication sessions.', {'entities': [[0, 3, 'PROTOCOL']]}],
    
    ['I saved the file as a PDF for easy sharing.', {'entities': [[20, 23, 'FILE_FORMAT']]}],
    
    ['JPEG is great for photos, but PNG is better for graphics.', {'entities': [[0, 4, 'FILE_FORMAT'], [25, 28, 'FILE_FORMAT']]}],
    
    ['Do you know how to convert a file to MP3?', {'entities': [[30, 33, 'FILE_FORMAT']]}],
    
    ['I prefer MP4 over AVI for video files.', {'entities': [[10, 13, 'FILE_FORMAT'], [19, 22, 'FILE_FORMAT']]}],
    
    ['The report is in DOCX format; can you open it?', {'entities': [[18, 22, 'FILE_FORMAT']]}],
    
    ['I need to export this data to CSV for analysis.', {'entities': [[28, 31, 'FILE_FORMAT']]}],
    
    ['JSON is my preferred format for APIs.', {'entities': [[0, 4, 'FILE_FORMAT']]}],
    
    ['XML is so verbose compared to JSON.', {'entities': [[0, 3, 'FILE_FORMAT'], [24, 28, 'FILE_FORMAT']]}],
    
    ['I’m not sure how to open a ZIP file.', {'entities': [[20, 23, 'FILE_FORMAT']]}],
    
    ['Can you send me the file in TXT format?', {'entities': [[29, 32, 'FILE_FORMAT']]}],
    
    ['I saved the image as a PNG for better quality.', {'entities': [[22, 25, 'FILE_FORMAT']]}],
    
    ['The video is in MKV format; can you play it?', {'entities': [[18, 21, 'FILE_FORMAT']]}],
    
    ['I need to convert this file to WAV for better audio quality.', {'entities': [[30, 33, 'FILE_FORMAT']]}],
    
    ['The spreadsheet is in XLSX format.', {'entities': [[23, 27, 'FILE_FORMAT']]}],
    
    ['I prefer using OGG for audio files.', {'entities': [[19, 22, 'FILE_FORMAT']]}],
    
    ['The presentation is in PPTX format.', {'entities': [[23, 27, 'FILE_FORMAT']]}],
    
    ['I need to save this file as a GIF for the animation.', {'entities': [[26, 29, 'FILE_FORMAT']]}],
    
    ['The database backup is in SQL format.', {'entities': [[24, 27, 'FILE_FORMAT']]}],
    
    ['I saved the document as an RTF for compatibility.', {'entities': [[26, 29, 'FILE_FORMAT']]}],
    
    ['The e-book is in EPUB format.', {'entities': [[17, 21, 'FILE_FORMAT']]}],
    
    ['Phishing attacks are becoming more sophisticated.', {'entities': [[0, 8, 'CYBERSECURITY_TERM']]}],
    
    ['I need to install antivirus software to protect against malware.', {'entities': [[48, 54, 'CYBERSECURITY_TERM']]}],
    
    ['Ransomware locked all my files until I paid the fee.', {'entities': [[0, 11, 'CYBERSECURITY_TERM']]}],
    
    ['A firewall is essential for network security.', {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    
    ['Encryption is the best way to secure sensitive data.', {'entities': [[0, 10, 'CYBERSECURITY_TERM']]}],
    
    ['Two-factor authentication adds an extra layer of security.', {'entities': [[0, 24, 'CYBERSECURITY_TERM']]}],
    
    ['I use a VPN to protect my online privacy.', {'entities': [[9, 12, 'CYBERSECURITY_TERM']]}],
    
    ['Zero-day exploits are hard to defend against.', {'entities': [[0, 14, 'CYBERSECURITY_TERM']]}],
    
    ['Social engineering is a common tactic for hackers.', {'entities': [[0, 18, 'CYBERSECURITY_TERM']]}],
    
    ['A DDoS attack can take down an entire website.', {'entities': [[2, 10, 'CYBERSECURITY_TERM']]}],
    
    ['A botnet is a network of infected devices.', {'entities': [[2, 8, 'CYBERSECURITY_TERM']]}],
    
    ['A keylogger records every keystroke you make.', {'entities': [[2, 11, 'CYBERSECURITY_TERM']]}],
    
    ['A trojan horse disguises itself as legitimate software.', {'entities': [[2, 15, 'CYBERSECURITY_TERM']]}],
    
    ['A worm spreads without user interaction.', {'entities': [[2, 6, 'CYBERSECURITY_TERM']]}],
    
    ['A rootkit provides unauthorized access to a system.', {'entities': [[2, 9, 'CYBERSECURITY_TERM']]}],
    
    ['Penetration testing identifies system vulnerabilities.', {'entities': [[0, 19, 'CYBERSECURITY_TERM']]}],
    
    ['A honeypot lures attackers to study their methods.', {'entities': [[2, 9, 'CYBERSECURITY_TERM']]}],
    
    ['A security patch fixes vulnerabilities in software.', {'entities': [[2, 16, 'CYBERSECURITY_TERM']]}],
    
    ['A vulnerability scanner detects weaknesses in a system.', {'entities': [[2, 23, 'CYBERSECURITY_TERM']]}],
    
    ['A brute force attack tries all possible passwords.', {'entities': [[2, 16, 'CYBERSECURITY_TERM']]}],
    
    ['I’ve been using Python and JavaScript for my web projects.', {'entities': [[17, 23, 'PROGRAMMING_LANGUAGE'], [28, 38, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Do you think Java or C++ is better for game development?', {'entities': [[13, 17, 'PROGRAMMING_LANGUAGE'], [21, 24, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I’m learning Swift for iOS and Kotlin for Android development.', {'entities': [[12, 17, 'PROGRAMMING_LANGUAGE'], [29, 35, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Rust is great for system programming, but Go is easier to learn.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE'], [41, 43, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I prefer Ruby for scripting and Python for data analysis.', {'entities': [[10, 14, 'PROGRAMMING_LANGUAGE'], [30, 36, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Have you tried TypeScript? It’s a superset of JavaScript.', {'entities': [[14, 24, 'PROGRAMMING_LANGUAGE'], [45, 55, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I use PHP for backend development and React for the frontend.', {'entities': [[8, 11, 'PROGRAMMING_LANGUAGE'], [35, 40, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Perl is great for text processing, but Python is more popular.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE'], [36, 42, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Scala is used for big data, and Java is still widely used.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE'], [31, 35, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Dart is the language behind Flutter, and it’s really powerful.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Lua is often used in game development, like in Roblox.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Julia is gaining traction in data science and machine learning.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['R is my go-to language for statistical analysis and visualization.', {'entities': [[13, 14, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Objective-C is still used in some legacy iOS projects.', {'entities': [[0, 11, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Bash is essential for scripting on Unix and Linux systems.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['PowerShell is great for Windows automation and scripting.', {'entities': [[0, 11, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Groovy is a dynamic language for the Java platform.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['F# is a functional-first language for .NET development.', {'entities': [[0, 2, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Haskell is purely functional, and it’s quite challenging.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Elixir is built on the Erlang virtual machine.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Clojure is a modern Lisp dialect for the JVM.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['D is a systems programming language with modern features.', {'entities': [[0, 1, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Nim is a statically typed compiled language.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Zig is a general-purpose programming language.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Crystal is inspired by Ruby and focuses on performance.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Elm is a functional language for front-end development.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['ReasonML is a syntax extension for OCaml.', {'entities': [[0, 8, 'PROGRAMMING_LANGUAGE']]}],
    
    ['PureScript is a strongly-typed functional language.', {'entities': [[0, 10, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Idris is a general-purpose functional language.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['ATS is a language with formal verification features.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Agda is a dependently typed functional language.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Coq is used for formal verification of software.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Isabelle is a proof assistant and programming language.', {'entities': [[0, 9, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Lean is a functional programming language.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['VHDL is used for hardware description and simulation.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Verilog is another hardware description language.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['ABAP is used for SAP application development.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Fortran is still used in scientific computing.', {'entities': [[0, 7, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Prolog is a logic programming language.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Erlang is known for its concurrency and fault tolerance.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['OCaml is a functional language with type inference.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Smalltalk is an object-oriented programming language.', {'entities': [[0, 9, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Scheme is a minimalist dialect of Lisp.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Racket is a general-purpose programming language.', {'entities': [[0, 6, 'PROGRAMMING_LANGUAGE']]}],
    
    ['COBOL is still used in legacy systems.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Ada is used in safety-critical systems.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Lisp is one of the oldest programming languages.', {'entities': [[0, 4, 'PROGRAMMING_LANGUAGE']]}],
    
    ['Forth is a stack-based programming language.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['APL is known for its concise syntax.', {'entities': [[0, 3, 'PROGRAMMING_LANGUAGE']]}],
    
    ['BASIC is a beginner-friendly programming language.', {'entities': [[0, 5, 'PROGRAMMING_LANGUAGE']]}],
    
    ['I use React for the frontend and Django for the backend.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [30, 36, 'FRAMEWORK_LIBRARY']]}],
    
    ['Angular is great for large-scale applications, but Vue.js is simpler.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [52, 58, 'FRAMEWORK_LIBRARY']]}],
    
    ['Flask is lightweight, while Spring is more robust.', {'entities': [[0, 5, 'FRAMEWORK_LIBRARY'], [25, 31, 'FRAMEWORK_LIBRARY']]}],
    
    ['I prefer TensorFlow for machine learning, but PyTorch is also good.', {'entities': [[10, 20, 'FRAMEWORK_LIBRARY'], [43, 50, 'FRAMEWORK_LIBRARY']]}],
    
    ['Bootstrap is great for responsive design, and Tailwind CSS is modern.', {'entities': [[0, 9, 'FRAMEWORK_LIBRARY'], [41, 53, 'FRAMEWORK_LIBRARY']]}],
    
    ['Laravel is my go-to PHP framework, but Symfony is also popular.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [37, 44, 'FRAMEWORK_LIBRARY']]}],
    
    ['Express.js is perfect for building APIs, and Nest.js is gaining traction.', {'entities': [[0, 10, 'FRAMEWORK_LIBRARY'], [44, 51, 'FRAMEWORK_LIBRARY']]}],
    
    ['Ruby on Rails is great for rapid development, and Flask is lightweight.', {'entities': [[0, 13, 'FRAMEWORK_LIBRARY'], [51, 56, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Keras for prototyping, but TensorFlow is more powerful.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [35, 45, 'FRAMEWORK_LIBRARY']]}],
    
    ['Next.js is great for server-side rendering, and Nuxt.js is for Vue.js.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [44, 51, 'FRAMEWORK_LIBRARY']]}],
    
    ['Flutter is amazing for cross-platform apps, and React Native is also good.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [49, 61, 'FRAMEWORK_LIBRARY']]}],
    
    ['Xamarin is great for C# developers, and Ionic is for hybrid apps.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [38, 43, 'FRAMEWORK_LIBRARY']]}],
    
    ['ASP.NET is powerful for web applications, and Django is for Python.', {'entities': [[0, 7, 'FRAMEWORK_LIBRARY'], [44, 50, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use jQuery for simple tasks, but React is better for complex UIs.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'], [35, 40, 'FRAMEWORK_LIBRARY']]}],
    
    ['Ember.js is great for ambitious apps, and Svelte is modern.', {'entities': [[0, 8, 'FRAMEWORK_LIBRARY'], [41, 47, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Flask for small projects, and Django for larger ones.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [33, 39, 'FRAMEWORK_LIBRARY']]}],
    
    ['Spring is great for Java, and Laravel is for PHP.', {'entities': [[0, 6, 'FRAMEWORK_LIBRARY'], [31, 38, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use TensorFlow for deep learning, and Scikit-learn for machine learning.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [47, 59, 'FRAMEWORK_LIBRARY']]}],
    
    ['I prefer Bootstrap for styling, but Material-UI is also good.', {'entities': [[10, 19, 'FRAMEWORK_LIBRARY'], [41, 52, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Express.js for APIs, and FastAPI is also great.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [36, 43, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Flask for small projects, and FastAPI for APIs.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [34, 41, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use React for the frontend, and Node.js for the backend.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [35, 42, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Django for web development, and Flask for microservices.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'], [41, 46, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use TensorFlow for deep learning, and Keras for prototyping.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [44, 49, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Angular for large apps, and Vue.js for smaller ones.', {'entities': [[8, 15, 'FRAMEWORK_LIBRARY'], [38, 44, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Spring for Java apps, and Flask for Python apps.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'], [35, 40, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Laravel for PHP apps, and Django for Python apps.', {'entities': [[8, 15, 'FRAMEWORK_LIBRARY'], [36, 42, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use React Native for mobile apps, and Flutter for cross-platform.', {'entities': [[8, 20, 'FRAMEWORK_LIBRARY'], [44, 51, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Bootstrap for styling, and Tailwind CSS for modern designs.', {'entities': [[8, 17, 'FRAMEWORK_LIBRARY'], [42, 54, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Express.js for APIs, and FastAPI for Python APIs.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [36, 43, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Flask for small projects, and Django for larger ones.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [33, 39, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use TensorFlow for deep learning, and Scikit-learn for machine learning.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [47, 59, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use React for the frontend, and Node.js for the backend.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [35, 42, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Django for web development, and Flask for microservices.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'],[41, 46, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use TensorFlow for deep learning, and Keras for prototyping.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [44, 49, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Angular for large apps, and Vue.js for smaller ones.', {'entities': [[8, 15, 'FRAMEWORK_LIBRARY'], [38, 44, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Spring for Java apps, and Flask for Python apps.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'], [35, 40, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Laravel for PHP apps, and Django for Python apps.', {'entities': [[8, 15, 'FRAMEWORK_LIBRARY'], [36, 42, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use React Native for mobile apps, and Flutter for cross-platform development.', {'entities': [[8, 20, 'FRAMEWORK_LIBRARY'], [44, 51, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Bootstrap for styling, and Tailwind CSS for modern designs.', {'entities': [[8, 17, 'FRAMEWORK_LIBRARY'], [42, 54, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Express.js for APIs, and FastAPI for Python APIs.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [36, 43, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Flask for small projects, and Django for larger ones.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [33, 39, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use TensorFlow for deep learning, and Scikit-learn for machine learning.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [47, 59, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use React for the frontend, and Node.js for the backend.', {'entities': [[8, 13, 'FRAMEWORK_LIBRARY'], [35, 42, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Django for web development, and Flask for microservices.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'], [41, 46, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use TensorFlow for deep learning, and Keras for prototyping.', {'entities': [[8, 18, 'FRAMEWORK_LIBRARY'], [44, 49, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Angular for large apps, and Vue.js for smaller ones.', {'entities': [[8, 15, 'FRAMEWORK_LIBRARY'], [38, 44, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Spring for Java apps, and Flask for Python apps.', {'entities': [[8, 14, 'FRAMEWORK_LIBRARY'], [35, 40, 'FRAMEWORK_LIBRARY']]}],
    
    ['I use Laravel for PHP apps, and Django for Python apps.', {'entities': [[8, 15, 'FRAMEWORK_LIBRARY'], [36, 42, 'FRAMEWORK_LIBRARY']]}],
    
    ['My CPU and GPU are both overheating; I need better cooling.', {'entities': [[3, 6, 'HARDWARE'], [11, 14, 'HARDWARE']]}],
    
    ['I upgraded my SSD, and now my computer is much faster.', {'entities': [[15, 18, 'HARDWARE']]}],
    
    ['The motherboard and RAM in my PC need an upgrade.', {'entities': [[4, 15, 'HARDWARE'], [20, 23, 'HARDWARE']]}],
    
    ['I bought a new keyboard and mouse for my home office.', {'entities': [[15, 23, 'HARDWARE'], [28, 33, 'HARDWARE']]}],
    
    ['My monitor and printer are both acting up today.', {'entities': [[3, 10, 'HARDWARE'], [15, 22, 'HARDWARE']]}],
    
    ['I need a new power supply unit and cooling fan for my PC.', {'entities': [[13, 29, 'HARDWARE'], [34, 45, 'HARDWARE']]}],
    
    ['The hard drive and graphics card in my laptop are outdated.', {'entities': [[4, 14, 'HARDWARE'], [19, 33, 'HARDWARE']]}],
    
    ['I’m thinking of getting a new sound card and network adapter.', {'entities': [[28, 38, 'HARDWARE'], [43, 58, 'HARDWARE']]}],
    
    ['My optical drive and USB flash drive are both broken.', {'entities': [[3, 16, 'HARDWARE'], [21, 35, 'HARDWARE']]}],
    
    ['I need a new case and heatsink for my gaming PC.', {'entities': [[13, 17, 'HARDWARE'], [22, 30, 'HARDWARE']]}],
    
    ['The microphone and webcam I bought are great for video calls.', {'entities': [[4, 14, 'HARDWARE'], [19, 25, 'HARDWARE']]}],
    
    ['I’m looking for a new router and modem for my home network.', {'entities': [[20, 26, 'HARDWARE'], [31, 36, 'HARDWARE']]}],
    
    ['My touchpad and stylus are both not working properly.', {'entities': [[3, 11, 'HARDWARE'], [16, 22, 'HARDWARE']]}],
    
    ['I need a new VR headset and gaming console for my setup.', {'entities': [[13, 23, 'HARDWARE'], [28, 42, 'HARDWARE']]}],
    
    ['The smartwatch and fitness tracker I use are both great.', {'entities': [[4, 14, 'HARDWARE'], [19, 35, 'HARDWARE']]}],
    
    ['I’m thinking of buying a drone and 3D printer for my projects.', {'entities': [[24, 29, 'HARDWARE'], [34, 44, 'HARDWARE']]}],
    
    ['My external hard drive and NAS device are both full.', {'entities': [[3, 20, 'HARDWARE'], [25, 35, 'HARDWARE']]}],
    
    ['I need a new graphics tablet and smart speaker for my office.', {'entities': [[13, 29, 'HARDWARE'], [34, 47, 'HARDWARE']]}],
    
    ['The biometric scanner and smart lock I installed are working well.', {'entities': [[4, 21, 'HARDWARE'], [26, 36, 'HARDWARE']]}],
    
    ['I’m looking for a new robotic vacuum and smart bulb for my home.', {'entities': [[20, 35, 'HARDWARE'], [40, 50, 'HARDWARE']]}],
    
    ['My gaming headset and thermal printer are both great.', {'entities': [[3, 17, 'HARDWARE'], [22, 38, 'HARDWARE']]}],
    
    ['I need a new point-of-sale system and barcode scanner for my store.', {'entities': [[13, 32, 'HARDWARE'], [37, 53, 'HARDWARE']]}],
    
    ['The server rack and docking station in my office need an upgrade.', {'entities': [[4, 16, 'HARDWARE'], [21, 36, 'HARDWARE']]}],
    
    ['I’m thinking of getting a new e-reader and smart thermostat.', {'entities': [[24, 32, 'HARDWARE'], [37, 53, 'HARDWARE']]}],
    
    ['My home assistant device and smart lock are both working well.', {'entities': [[3, 24, 'HARDWARE'], [29, 39, 'HARDWARE']]}],
    
    ['I need a new gaming console and VR headset for my setup.', {'entities': [[13, 27, 'HARDWARE'], [32, 42, 'HARDWARE']]}],
    
    ['The drone and 3D printer I bought are both amazing.', {'entities': [[4, 9, 'HARDWARE'], [14, 24, 'HARDWARE']]}],
    
    ['I’m looking for a new smartwatch and fitness tracker.', {'entities': [[20, 30, 'HARDWARE'], [35, 51, 'HARDWARE']]}],
    
    ['My external hard drive and USB flash drive are both full.', {'entities': [[3, 20, 'HARDWARE'], [25, 39, 'HARDWARE']]}],
    
    ['I need a new graphics tablet and smart speaker for my office.', {'entities': [[13, 29, 'HARDWARE'], [34, 47, 'HARDWARE']]}],
    
    ['The biometric scanner and smart lock I installed are working well.', {'entities': [[4, 21, 'HARDWARE'], [26, 36, 'HARDWARE']]}],
    
    ['I’m looking for a new robotic vacuum and smart bulb for my home.', {'entities': [[20, 35, 'HARDWARE'], [40, 50, 'HARDWARE']]}],
    
    ['My gaming headset and thermal printer are both great.', {'entities': [[3, 17, 'HARDWARE'], [22, 38, 'HARDWARE']]}],
    
    ['I need a new point-of-sale system and barcode scanner for my store.', {'entities': [[13, 32, 'HARDWARE'], [37, 53, 'HARDWARE']]}],
    
    ['The server rack and docking station in my office need an upgrade.', {'entities': [[4, 16, 'HARDWARE'], [21, 36, 'HARDWARE']]}],
    
    ['I’m thinking of getting a new e-reader and smart thermostat.', {'entities': [[24, 32, 'HARDWARE'], [37, 53, 'HARDWARE']]}],
    
    ['My home assistant device and smart lock are both working well.', {'entities': [[3, 24, 'HARDWARE'], [29, 39, 'HARDWARE']]}],
    
    ['I need a new gaming console and VR headset for my setup.', {'entities': [[13, 27, 'HARDWARE'], [32, 42, 'HARDWARE']]}],
    
    ['The drone and 3D printer I bought are both amazing.', {'entities': [[4, 9, 'HARDWARE'], [14, 24, 'HARDWARE']]}],
    
    ['I’m looking for a new smartwatch and fitness tracker.', {'entities': [[20, 30, 'HARDWARE'], [35, 51, 'HARDWARE']]}],
    
    ['My external hard drive and USB flash drive are both full.', {'entities': [[3, 20, 'HARDWARE'], [25, 39, 'HARDWARE']]}],
    
    ['I need a new graphics tablet and smart speaker for my office.', {'entities': [[13, 29, 'HARDWARE'], [34, 47, 'HARDWARE']]}],
    
    ['The biometric scanner and smart lock I installed are working well.', {'entities': [[4, 21, 'HARDWARE'], [26, 36, 'HARDWARE']]}],
    
    ['I’m looking for a new robotic vacuum and smart bulb for my home.', {'entities': [[20, 35, 'HARDWARE'], [40, 50, 'HARDWARE']]}],
    
    ['My gaming headset and thermal printer are both great.', {'entities': [[3, 17, 'HARDWARE'], [22, 38, 'HARDWARE']]}],
    
    ['I need a new point-of-sale system and barcode scanner for my store.', {'entities': [[13, 32, 'HARDWARE'], [37, 53, 'HARDWARE']]}],
    
    ['The server rack and docking station in my office need an upgrade.', {'entities': [[4, 16, 'HARDWARE'], [21, 36, 'HARDWARE']]}],
    
    ['I’m thinking of getting a new e-reader and smart thermostat.', {'entities': [[24, 32, 'HARDWARE'], [37, 53, 'HARDWARE']]}],
    
    ['My home assistant device and smart lock are both working well.', {'entities': [[3, 24, 'HARDWARE'],[29, 39, 'HARDWARE']]}],
    
    ['I need a new gaming console and VR headset for my setup.', {'entities': [[13, 27, 'HARDWARE'], [32, 42, 'HARDWARE']]}],
    
    ['I’m using k-means for clustering and linear regression for prediction.', {'entities': [[11, 18, 'ALGORITHM_MODEL'], [43, 59, 'ALGORITHM_MODEL']]}],
    
    ['Decision trees are great for classification, and random forest is even better.', {'entities': [[0, 14, 'ALGORITHM_MODEL'], [49, 62, 'ALGORITHM_MODEL']]}],
    
    ['Support vector machines are powerful, but neural networks are more flexible.', {'entities': [[0, 22, 'ALGORITHM_MODEL'], [44, 59, 'ALGORITHM_MODEL']]}],
    
    ['Gradient boosting is used in competitions, and XGBoost is a popular implementation.', {'entities': [[0, 18, 'ALGORITHM_MODEL'], [49, 56, 'ALGORITHM_MODEL']]}],
    
    ['K-nearest neighbors is simple, but principal component analysis is more complex.', {'entities': [[0, 19, 'ALGORITHM_MODEL'], [40, 66, 'ALGORITHM_MODEL']]}],
    
    ['Apriori is great for market basket analysis, and Naive Bayes is good for classification.', {'entities': [[0, 7, 'ALGORITHM_MODEL'], [50, 61, 'ALGORITHM_MODEL']]}],
    

    ['PageRank is the algorithm behind Google, and logistic regression is used for binary classification.', {'entities': [[0, 9, 'ALGORITHM_MODEL'], [44, 63, 'ALGORITHM_MODEL']]}],['AdaBoost is an ensemble method, and hidden Markov models are used for sequence prediction.', {'entities': [[0, 9, 'ALGORITHM_MODEL'], [40, 60, 'ALGORITHM_MODEL']]}],

    ['DBSCAN is great for density-based clustering, and t-SNE is used for visualization.', {'entities': [[0, 7, 'ALGORITHM_MODEL'], [52, 58, 'ALGORITHM_MODEL']]}],

    ['The EM algorithm is used for parameter estimation, and Q-learning is used in reinforcement learning.', {'entities': [[4, 15, 'ALGORITHM_MODEL'], [54, 64, 'ALGORITHM_MODEL']]}],

    ['The Perceptron is a simple neural network, and LSTM is used for sequential data.', {'entities': [[4, 14, 'ALGORITHM_MODEL'], [46, 50, 'ALGORITHM_MODEL']]}],

    ['GANs are used for generating data, and CNNs are great for image processing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['RNNs are used for time series, and BERT is great for natural language processing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['YOLO is used for object detection, and GPT is great for text generation.', {'entities': [[0, 5, 'ALGORITHM_MODEL'], [42, 45, 'ALGORITHM_MODEL']]}],

    ['SVM is great for classification, and KNN is simple and effective.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['PCA is used for dimensionality reduction, and ARIMA is great for time series forecasting.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [49, 55, 'ALGORITHM_MODEL']]}],

    ['Markov chains are used for prediction, and Monte Carlo methods are used for simulation.', {'entities': [[0, 13, 'ALGORITHM_MODEL'], [48, 61, 'ALGORITHM_MODEL']]}],

    ['Dijkstra’s algorithm finds the shortest path, and Floyd-Warshall solves all-pairs shortest paths.', {'entities': [[0, 18, 'ALGORITHM_MODEL'], [55, 71, 'ALGORITHM_MODEL']]}],

    ['Bellman-Ford handles negative weights, and Kruskal’s algorithm finds minimum spanning trees.', {'entities': [[0, 13, 'ALGORITHM_MODEL'], [50, 66, 'ALGORITHM_MODEL']]}],

    ['Prim’s algorithm is another approach for minimum spanning trees, and A* is used for pathfinding.', {'entities': [[0, 12, 'ALGORITHM_MODEL'], [63, 66, 'ALGORITHM_MODEL']]}],

    ['RSA is used for encryption, and AES is a symmetric encryption standard.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['DES is an older encryption method, and SHA is used for hashing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [41, 45, 'ALGORITHM_MODEL']]}],

    ['ElGamal is used for public-key cryptography, and Diffie-Hellman enables secure key exchange.', {'entities': [[0, 8, 'ALGORITHM_MODEL'], [53, 68, 'ALGORITHM_MODEL']]}],

    ['ECC is used for elliptic curve cryptography, and Simplex is used for linear programming.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [53, 61, 'ALGORITHM_MODEL']]}],

    ['The Viterbi algorithm is used for decoding, and the EM algorithm is used for parameter estimation.', {'entities': [[4, 20, 'ALGORITHM_MODEL'], [55, 66, 'ALGORITHM_MODEL']]}],

    ['The Perceptron is a simple neural network, and LSTM is used for sequential data.', {'entities': [[4, 14, 'ALGORITHM_MODEL'], [46, 50, 'ALGORITHM_MODEL']]}],

    ['GANs are used for generating data, and CNNs are great for image processing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['RNNs are used for time series, and BERT is great for natural language processing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['YOLO is used for object detection, and GPT is great for text generation.', {'entities': [[0, 5, 'ALGORITHM_MODEL'], [42, 45, 'ALGORITHM_MODEL']]}],

    ['SVM is great for classification, and KNN is simple and effective.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['PCA is used for dimensionality reduction, and ARIMA is great for time series forecasting.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [49, 55, 'ALGORITHM_MODEL']]}],

    ['Markov chains are used for prediction, and Monte Carlo methods are used for simulation.', {'entities': [[0, 13, 'ALGORITHM_MODEL'], [48, 61, 'ALGORITHM_MODEL']]}],

    ['Dijkstra’s algorithm finds the shortest path, and Floyd-Warshall solves all-pairs shortest paths.', {'entities': [[0, 18, 'ALGORITHM_MODEL'], [55, 71, 'ALGORITHM_MODEL']]}],

    ['Bellman-Ford handles negative weights, and Kruskal’s algorithm finds minimum spanning trees.', {'entities': [[0, 13, 'ALGORITHM_MODEL'], [50, 66, 'ALGORITHM_MODEL']]}],

    ['Prim’s algorithm is another approach for minimum spanning trees, and A* is used for pathfinding.', {'entities': [[0, 12, 'ALGORITHM_MODEL'], [63, 66, 'ALGORITHM_MODEL']]}],

    ['RSA is used for encryption, and AES is a symmetric encryption standard.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['DES is an older encryption method, and SHA is used for hashing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [41, 45, 'ALGORITHM_MODEL']]}],

    ['ElGamal is used for public-key cryptography, and Diffie-Hellman enables secure key exchange.', {'entities': [[0, 8, 'ALGORITHM_MODEL'], [53, 68, 'ALGORITHM_MODEL']]}],

    ['ECC is used for elliptic curve cryptography, and Simplex is used for linear programming.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [53, 61, 'ALGORITHM_MODEL']]}],

    ['The Viterbi algorithm is used for decoding, and the EM algorithm is used for parameter estimation.', {'entities': [[4, 20, 'ALGORITHM_MODEL'], [55, 66, 'ALGORITHM_MODEL']]}],

    ['The Perceptron is a simple neural network, and LSTM is used for sequential data.', {'entities': [[4, 14, 'ALGORITHM_MODEL'], [46, 50, 'ALGORITHM_MODEL']]}],

    ['GANs are used for generating data, and CNNs are great for image processing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['RNNs are used for time series, and BERT is great for natural language processing.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['YOLO is used for object detection, and GPT is great for text generation.', {'entities': [[0, 5, 'ALGORITHM_MODEL'], [42, 45, 'ALGORITHM_MODEL']]}],

    ['SVM is great for classification, and KNN is simple and effective.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [39, 43, 'ALGORITHM_MODEL']]}],

    ['PCA is used for dimensionality reduction, and ARIMA is great for time series forecasting.', {'entities': [[0, 4, 'ALGORITHM_MODEL'], [49, 55, 'ALGORITHM_MODEL']]}],

    ['Markov chains are used for prediction, and Monte Carlo methods are used for simulation.', {'entities': [[0, 13, 'ALGORITHM_MODEL'], [48, 61, 'ALGORITHM_MODEL']]}],

    ['Dijkstra’s algorithm finds the shortest path, and Floyd-Warshall solves all-pairs shortest paths.', {'entities': [[0, 18, 'ALGORITHM_MODEL'], [55, 71, 'ALGORITHM_MODEL']]}],

    ['Bellman-Ford handles negative weights, and Kruskal’s algorithm finds minimum spanning trees.', {'entities': [[0, 13, 'ALGORITHM_MODEL'], [50, 66, 'ALGORITHM_MODEL']]}],

    ['Prim’s algorithm is another approach for minimum spanning trees, and A* is used for pathfinding.', {'entities': [[0, 12, 'ALGORITHM_MODEL'], [63, 66, 'ALGORITHM_MODEL']]}],
    

    ['HTTP is used for web communication, and HTTPS is more secure.', {'entities': [[0, 4, 'PROTOCOL'], [39, 44, 'PROTOCOL']]}],
    ['FTP is great for file transfers, but SFTP is more secure.', {'entities': [[0, 3, 'PROTOCOL'], [37, 41, 'PROTOCOL']]}],

    ['SMTP is used for sending emails, and POP3 is for receiving them.', {'entities': [[0, 4, 'PROTOCOL'], [38, 42, 'PROTOCOL']]}],

    ['IMAP is better than POP3 for managing emails on a server.', {'entities': [[0, 4, 'PROTOCOL'], [20, 24, 'PROTOCOL']]}],

    ['TCP ensures reliable data delivery, and UDP is faster but less reliable.', {'entities': [[0, 3, 'PROTOCOL'], [41, 44, 'PROTOCOL']]}],

    ['DNS translates domain names, and DHCP assigns IP addresses.', {'entities': [[0, 3, 'PROTOCOL'], [37, 41, 'PROTOCOL']]}],

    ['SSH is used for secure remote access, and Telnet is outdated.', {'entities': [[0, 3, 'PROTOCOL'], [41, 47, 'PROTOCOL']]}],

    ['RDP is great for remote desktop connections, and VNC is another option.', {'entities': [[0, 3, 'PROTOCOL'], [52, 55, 'PROTOCOL']]}],

    ['SNMP is used for network management, and NTP synchronizes clocks.', {'entities': [[0, 4, 'PROTOCOL'], [43, 46, 'PROTOCOL']]}],

    ['ICMP is used for error reporting, and ARP resolves IP addresses.', {'entities': [[0, 4, 'PROTOCOL'], [41, 44, 'PROTOCOL']]}],

    ['RTP is used for audio and video streaming, and RTSP controls the streams.', {'entities': [[0, 3, 'PROTOCOL'], [52, 56, 'PROTOCOL']]}],

    ['SIP is used for initiating communication sessions, and WebSocket enables real-time communication.', {'entities': [[0, 3, 'PROTOCOL'], [58, 68, 'PROTOCOL']]}],

    ['LDAP is used for directory services, and BGP is a routing protocol.', {'entities': [[0, 4, 'PROTOCOL'], [44, 47, 'PROTOCOL']]}],

    ['OSPF is a routing protocol, and EIGRP is Cisco’s proprietary protocol.', {'entities': [[0, 4, 'PROTOCOL'], [38, 43, 'PROTOCOL']]}],

    ['RIP is a simple routing protocol, and IPsec provides secure communication.', {'entities': [[0, 3, 'PROTOCOL'], [44, 49, 'PROTOCOL']]}],

    ['SSL ensures secure communication, and TLS is its successor.', {'entities': [[0, 3, 'PROTOCOL'], [41, 44, 'PROTOCOL']]}],

    ['SFTP is more secure than FTP, and TFTP is a simpler version.', {'entities': [[0, 4, 'PROTOCOL'], [25, 28, 'PROTOCOL'], [49, 53, 'PROTOCOL']]}],

    ['MQTT is a lightweight messaging protocol, and CoAP is for constrained devices.', {'entities': [[0, 4, 'PROTOCOL'], [53, 57, 'PROTOCOL']]}],

    ['HTTP/2 improves web performance, and QUIC is a modern transport protocol.', {'entities': [[0, 6, 'PROTOCOL'], [44, 48, 'PROTOCOL']]}],

    ['SMB is used for file sharing, and NFS is for Unix systems.', {'entities': [[0, 3, 'PROTOCOL'], [38, 41, 'PROTOCOL']]}],

    ['AFP is used for file sharing in macOS, and NetBIOS is an older protocol.', {'entities': [[0, 3, 'PROTOCOL'], [44, 51, 'PROTOCOL']]}],

    ['PPTP is a VPN protocol, and L2TP is more secure.', {'entities': [[0, 4, 'PROTOCOL'], [36, 40, 'PROTOCOL']]}],

    ['OpenVPN is an open-source VPN protocol, and IKE is used for key exchange.', {'entities': [[0, 8, 'PROTOCOL'], [52, 55, 'PROTOCOL']]}],
    

    ['GRE is a tunneling protocol, and STP prevents loops in networks.', {'entities': [[0, 3, 'PROTOCOL'], [38, 41, 'PROTOCOL']]}],
    ['VLAN is used for network segmentation, and CDP is a Cisco discovery protocol.', {'entities': [[0, 4, 'PROTOCOL'], [48, 51, 'PROTOCOL']]}],

    ['LLDP is a vendor-neutral protocol, and RADIUS is used for network authentication.', {'entities': [[0, 4, 'PROTOCOL'], [46, 52, 'PROTOCOL']]}],

    ['TACACS+ is a Cisco authentication protocol, and Kerberos is used for secure authentication.', {'entities': [[0, 8, 'PROTOCOL'], [55, 63, 'PROTOCOL']]}],

    ['IPsec is used for secure communication, and OpenVPN is an open-source alternative.', {'entities': [[0, 5, 'PROTOCOL'], [49, 57, 'PROTOCOL']]}],

    ['HTTP/3 is the latest version of HTTP, and it uses QUIC for faster communication.', {'entities': [[0, 6, 'PROTOCOL'], [52, 56, 'PROTOCOL']]}],

    ['WebSocket enables real-time communication, and MQTT is great for IoT devices.', {'entities': [[0, 10, 'PROTOCOL'], [49, 53, 'PROTOCOL']]}],

    ['CoAP is designed for constrained devices, and AMQP is used for message queuing.', {'entities': [[0, 4, 'PROTOCOL'], [52, 56, 'PROTOCOL']]}],

    ['RADIUS is used for network authentication, and Diameter is its successor.', {'entities': [[0, 6, 'PROTOCOL'], [54, 62, 'PROTOCOL']]}],

    ['SNMP is used for network monitoring, and NetFlow is for traffic analysis.', {'entities': [[0, 4, 'PROTOCOL'], [46, 53, 'PROTOCOL']]}],

    ['FTP is used for file transfers, but FTPS is more secure.', {'entities': [[0, 3, 'PROTOCOL'], [37, 41, 'PROTOCOL']]}],

    ['Telnet is an older protocol, and SSH is the modern alternative.', {'entities': [[0, 6, 'PROTOCOL'], [41, 44, 'PROTOCOL']]}],

    ['RDP is great for remote desktop access, and VNC is another option.', {'entities': [[0, 3, 'PROTOCOL'], [52, 55, 'PROTOCOL']]}],

    ['DNS is essential for domain resolution, and DNSSEC adds security.', {'entities': [[0, 3, 'PROTOCOL'], [46, 52, 'PROTOCOL']]}],

    ['DHCP assigns IP addresses, and BOOTP is its predecessor.', {'entities': [[0, 4, 'PROTOCOL'], [38, 43, 'PROTOCOL']]}],

    ['ICMP is used for error reporting, and IGMP is for multicast communication.', {'entities': [[0, 4, 'PROTOCOL'], [41, 45, 'PROTOCOL']]}],

    ['RTP is used for real-time communication, and SRTP adds encryption.', {'entities': [[0, 3, 'PROTOCOL'], [46, 50, 'PROTOCOL']]}],

    ['SIP is used for VoIP, and H.323 is another VoIP protocol.', {'entities': [[0, 3, 'PROTOCOL'], [36, 41, 'PROTOCOL']]}],

    ['LDAP is used for directory services, and AD is Microsoft’s implementation.', {'entities': [[0, 4, 'PROTOCOL'], [44, 46, 'PROTOCOL']]}],

    ['BGP is used for routing between autonomous systems, and OSPF is for internal routing.', {'entities': [[0, 3, 'PROTOCOL'], [58, 62, 'PROTOCOL']]}],

    ['EIGRP is a Cisco routing protocol, and IS-IS is another option.', {'entities': [[0, 5, 'PROTOCOL'], [48, 53, 'PROTOCOL']]}],

    ['RIP is a simple routing protocol, and EIGRP is more advanced.', {'entities': [[0, 3, 'PROTOCOL'], [44, 49, 'PROTOCOL']]}],

    ['IPsec is used for secure communication, and GRE is for tunneling.', {'entities': [[0, 5, 'PROTOCOL'], [49, 52, 'PROTOCOL']]}],

    ['SSL is outdated, and TLS is the modern standard.', {'entities': [[0, 3, 'PROTOCOL'], [29, 32, 'PROTOCOL']]}],

    ['SFTP is more secure than FTP, and SCP is another secure option.', {'entities': [[0, 4, 'PROTOCOL'], [25, 28, 'PROTOCOL'], [53, 56, 'PROTOCOL']]}],

    ['MQTT is great for IoT, and AMQP is used for enterprise messaging.', {'entities': [[0, 4, 'PROTOCOL'], [41, 45, 'PROTOCOL']]}],

    ['HTTP/2 improves performance, and HTTP/3 uses QUIC for even faster communication.', {'entities': [[0, 6, 'PROTOCOL'], [42, 48, 'PROTOCOL'], [57, 61, 'PROTOCOL']]}],


    ['I saved the document as a PDF, but I also have a DOCX version.', {'entities': [[25, 28, 'FILE_FORMAT'], [51, 55, 'FILE_FORMAT']]}],
    ['JPEG is great for photos, but PNG is better for graphics.', {'entities': [[0, 4, 'FILE_FORMAT'], [30, 33, 'FILE_FORMAT']]}],

    ['MP3 is a common audio format, but WAV is lossless.', {'entities': [[0, 3, 'FILE_FORMAT'], [36, 39, 'FILE_FORMAT']]}],

    ['MP4 is a popular video format, but AVI is also widely used.', {'entities': [[0, 3, 'FILE_FORMAT'], [37, 40, 'FILE_FORMAT']]}],

    ['I prefer CSV for data storage, but JSON is better for APIs.', {'entities': [[12, 15, 'FILE_FORMAT'], [40, 44, 'FILE_FORMAT']]}],

    ['XML is verbose, but JSON is more lightweight.', {'entities': [[0, 3, 'FILE_FORMAT'], [24, 28, 'FILE_FORMAT']]}],

    ['HTML is used for web pages, and CSS is for styling.', {'entities': [[0, 4, 'FILE_FORMAT'], [33, 36, 'FILE_FORMAT']]}],

    ['ZIP is great for compression, but RAR is also popular.', {'entities': [[0, 3, 'FILE_FORMAT'], [37, 40, 'FILE_FORMAT']]}],

    ['GIF is great for animations, but WEBM is better for videos.', {'entities': [[0, 3, 'FILE_FORMAT'], [37, 41, 'FILE_FORMAT']]}],

    ['I saved the image as a BMP, but TIFF is higher quality.', {'entities': [[22, 25, 'FILE_FORMAT'], [39, 43, 'FILE_FORMAT']]}],

    ['SVG is great for vector graphics, and EPS is another option.', {'entities': [[0, 3, 'FILE_FORMAT'], [44, 47, 'FILE_FORMAT']]}],

    ['FLAC is a lossless audio format, but MP3 is more compact.', {'entities': [[0, 4, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['MKV is a versatile video format, and MOV is used by Apple.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['OGG is an open multimedia format, and WEBM is also open.', {'entities': [[0, 3, 'FILE_FORMAT'], [44, 48, 'FILE_FORMAT']]}],

    ['ISO is used for disk images, and DMG is for macOS.', {'entities': [[0, 3, 'FILE_FORMAT'], [35, 38, 'FILE_FORMAT']]}],

    ['EXE is an executable format, and APK is for Android apps.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['PY is the file extension for Python scripts, and JAR is for Java.', {'entities': [[0, 2, 'FILE_FORMAT'], [52, 55, 'FILE_FORMAT']]}],

    ['SQL is used for database scripts, and LOG is for logs.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['INI is a configuration file format, and YAML is more modern.', {'entities': [[0, 3, 'FILE_FORMAT'], [44, 48, 'FILE_FORMAT']]}],

    ['RTF is a text format with formatting, and TXT is plain text.', {'entities': [[0, 3, 'FILE_FORMAT'], [46, 49, 'FILE_FORMAT']]}],

    ['ODT is an open document format, and DOCX is more common.', {'entities': [[0, 3, 'FILE_FORMAT'], [43, 47, 'FILE_FORMAT']]}],

    ['ODS is for spreadsheets, and XLSX is the Excel format.', {'entities': [[0, 3, 'FILE_FORMAT'], [35, 39, 'FILE_FORMAT']]}],

    ['ODP is for presentations, and PPTX is the PowerPoint format.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 45, 'FILE_FORMAT']]}],

    ['EPUB is an e-book format, and MOBI is another option.', {'entities': [[0, 4, 'FILE_FORMAT'], [41, 45, 'FILE_FORMAT']]}],

    ['PSD is the Photoshop format, and AI is for Illustrator.', {'entities': [[0, 3, 'FILE_FORMAT'],[41, 43, 'FILE_FORMAT']]}],

    ['DWG is a CAD file format, and STL is for 3D models.', {'entities': [[0, 3, 'FILE_FORMAT'], [38, 41, 'FILE_FORMAT']]}],

    ['FBX is a 3D model format, and OBJ is another option.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['CSV is great for data storage, but JSON is better for APIs.', {'entities': [[0, 3, 'FILE_FORMAT'], [40, 44, 'FILE_FORMAT']]}],

    ['XML is verbose, but JSON is more lightweight.', {'entities': [[0, 3, 'FILE_FORMAT'], [24, 28, 'FILE_FORMAT']]}],

    ['HTML is used for web pages, and CSS is for styling.', {'entities': [[0, 4, 'FILE_FORMAT'], [33, 36, 'FILE_FORMAT']]}],

    ['ZIP is great for compression, but RAR is also popular.', {'entities': [[0, 3, 'FILE_FORMAT'], [37, 40, 'FILE_FORMAT']]}],

    ['GIF is great for animations, but WEBM is better for videos.', {'entities': [[0, 3, 'FILE_FORMAT'], [37, 41, 'FILE_FORMAT']]}],

    ['I saved the image as a BMP, but TIFF is higher quality.', {'entities': [[22, 25, 'FILE_FORMAT'], [39, 43, 'FILE_FORMAT']]}],

    ['SVG is great for vector graphics, and EPS is another option.', {'entities': [[0, 3, 'FILE_FORMAT'], [44, 47, 'FILE_FORMAT']]}],

    ['FLAC is a lossless audio format, but MP3 is more compact.', {'entities': [[0, 4, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['MKV is a versatile video format, and MOV is used by Apple.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['OGG is an open multimedia format, and WEBM is also open.', {'entities': [[0, 3, 'FILE_FORMAT'], [44, 48, 'FILE_FORMAT']]}],

    ['ISO is used for disk images, and DMG is for macOS.', {'entities': [[0, 3, 'FILE_FORMAT'], [35, 38, 'FILE_FORMAT']]}],

    ['EXE is an executable format, and APK is for Android apps.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['PY is the file extension for Python scripts, and JAR is for Java.', {'entities': [[0, 2, 'FILE_FORMAT'], [52, 55, 'FILE_FORMAT']]}],

    ['SQL is used for database scripts, and LOG is for logs.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['INI is a configuration file format, and YAML is more modern.', {'entities': [[0, 3, 'FILE_FORMAT'], [44, 48, 'FILE_FORMAT']]}],

    ['RTF is a text format with formatting, and TXT is plain text.', {'entities': [[0, 3, 'FILE_FORMAT'], [46, 49, 'FILE_FORMAT']]}],

    ['ODT is an open document format, and DOCX is more common.', {'entities': [[0, 3, 'FILE_FORMAT'], [43, 47, 'FILE_FORMAT']]}],

    ['ODS is for spreadsheets, and XLSX is the Excel format.', {'entities': [[0, 3, 'FILE_FORMAT'], [35, 39, 'FILE_FORMAT']]}],

    ['ODP is for presentations, and PPTX is the PowerPoint format.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 45, 'FILE_FORMAT']]}],

    ['EPUB is an e-book format, and MOBI is another option.', {'entities': [[0, 4, 'FILE_FORMAT'], [41, 45, 'FILE_FORMAT']]}],

    ['PSD is the Photoshop format, and AI is for Illustrator.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 43, 'FILE_FORMAT']]}],

    ['DWG is a CAD file format, and STL is for 3D models.', {'entities': [[0, 3, 'FILE_FORMAT'], [38, 41, 'FILE_FORMAT']]}],

    ['FBX is a 3D model format, and OBJ is another option.', {'entities': [[0, 3, 'FILE_FORMAT'], [41, 44, 'FILE_FORMAT']]}],

    ['Phishing attacks are common, and malware is a major threat.', {'entities': [[0, 8, 'CYBERSECURITY_TERM'], [34, 40, 'CYBERSECURITY_TERM']]}],

    ['Ransomware encrypts files, and a firewall protects networks.', {'entities': [[0, 11, 'CYBERSECURITY_TERM'], [40, 48, 'CYBERSECURITY_TERM']]}],

    ['Encryption ensures data security, and two-factor authentication adds an extra layer.', {'entities': [[0, 10, 'CYBERSECURITY_TERM'], [42, 66, 'CYBERSECURITY_TERM']]}],

    ['A VPN protects your privacy, and a zero-day exploit is hard to detect.', {'entities': [[2, 5, 'CYBERSECURITY_TERM'], [37, 51, 'CYBERSECURITY_TERM']]}],

    ['Social engineering manipulates users, and a DDoS attack overwhelms servers.', {'entities': [[0, 18, 'CYBERSECURITY_TERM'], [52, 60, 'CYBERSECURITY_TERM']]}],

    ['A botnet is a network of infected devices, and a keylogger records keystrokes.', {'entities': [[2, 8, 'CYBERSECURITY_TERM'], [55, 64, 'CYBERSECURITY_TERM']]}],

    ['A trojan horse disguises itself as legitimate software, and a worm spreads automatically.', {'entities': [[2, 15, 'CYBERSECURITY_TERM'], [66, 70, 'CYBERSECURITY_TERM']]}],

    ['A rootkit provides unauthorized access, and a honeypot lures attackers.', {'entities': [[2, 9, 'CYBERSECURITY_TERM'], [52, 59, 'CYBERSECURITY_TERM']]}],

    ['Penetration testing identifies vulnerabilities, and a security patch fixes them.', {'entities': [[0, 19, 'CYBERSECURITY_TERM'], [61, 75, 'CYBERSECURITY_TERM']]}],

    ['A vulnerability scanner detects weaknesses, and a brute force attack tries all passwords.', {'entities': [[2, 23, 'CYBERSECURITY_TERM'], [57, 71, 'CYBERSECURITY_TERM']]}],

    ['A man-in-the-middle attack intercepts communication, and SQL injection exploits databases.', {'entities': [[2, 24, 'CYBERSECURITY_TERM'], [66, 79, 'CYBERSECURITY_TERM']]}],

    ['Cross-site scripting targets web applications, and a data breach exposes sensitive information.', {'entities': [[0, 21, 'CYBERSECURITY_TERM'], [56, 67, 'CYBERSECURITY_TERM']]}],

    ['A security audit evaluates defenses, and a security policy defines rules.', {'entities': [[2, 16, 'CYBERSECURITY_TERM'], [50, 65, 'CYBERSECURITY_TERM']]}],

    ['A security incident requires immediate response, and a security token provides authentication.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [63, 77, 'CYBERSECURITY_TERM']]}],

    ['A security certificate ensures secure communication, and a security protocol defines rules.', {'entities': [[2, 21, 'CYBERSECURITY_TERM'], [63, 79, 'CYBERSECURITY_TERM']]}],

    ['A security framework provides guidelines, and a security control mitigates risks.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [53, 68, 'CYBERSECURITY_TERM']]}],

    ['A security risk assessment identifies threats, and a security breach compromises data.', {'entities': [[2, 25, 'CYBERSECURITY_TERM'], [60, 74, 'CYBERSECURITY_TERM']]}],

    ['A security vulnerability exposes systems, and a security threat is a potential danger.', {'entities': [[2, 23, 'CYBERSECURITY_TERM'], [56, 71, 'CYBERSECURITY_TERM']]}],

    ['A security measure protects against attacks, and a security awareness program educates users.', {'entities': [[2, 18, 'CYBERSECURITY_TERM'], [60, 83, 'CYBERSECURITY_TERM']]}],

    ['A security analyst monitors activity, and a security engineer designs secure systems.', {'entities': [[2, 18, 'CYBERSECURITY_TERM'], [53, 69, 'CYBERSECURITY_TERM']]}],

    ['A security consultant provides advice, and a security administrator manages defenses.', {'entities': [[2, 20, 'CYBERSECURITY_TERM'], [55, 76, 'CYBERSECURITY_TERM']]}],

    ['A security architect designs infrastructures, and a security operations center monitors threats.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [60, 86, 'CYBERSECURITY_TERM']]}],

    ['A security information and event management system collects logs, and a security baseline defines standards.', {'entities': [[2, 50, 'CYBERSECURITY_TERM'], [85, 101, 'CYBERSECURITY_TERM']]}],

    ['A security clearance grants access, and a security violation breaches policies.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [53, 70, 'CYBERSECURITY_TERM']]}],

    ['A security posture reflects defenses, and a security culture promotes awareness.', {'entities': [[2, 18, 'CYBERSECURITY_TERM'], [52, 67, 'CYBERSECURITY_TERM']]}],

    ['A phishing attack tricks users, and malware infects systems.', {'entities': [[2, 16, 'CYBERSECURITY_TERM'], [41, 47, 'CYBERSECURITY_TERM']]}],

    ['Ransomware locks files, and a firewall blocks unauthorized access.', {'entities': [[0, 11, 'CYBERSECURITY_TERM'], [37, 45, 'CYBERSECURITY_TERM']]}],

    ['Encryption secures data, and two-factor authentication adds security.', {'entities': [[0, 10, 'CYBERSECURITY_TERM'], [42, 66, 'CYBERSECURITY_TERM']]}],

    ['A VPN protects privacy, and a zero-day exploit is hard to detect.', {'entities': [[2, 5, 'CYBERSECURITY_TERM'], [37, 51, 'CYBERSECURITY_TERM']]}],

    ['Social engineering manipulates users, and a DDoS attack overwhelms servers.', {'entities': [[0, 18, 'CYBERSECURITY_TERM'], [52, 60, 'CYBERSECURITY_TERM']]}],

    ['A botnet is a network of infected devices, and a keylogger records keystrokes.', {'entities': [[2, 8, 'CYBERSECURITY_TERM'], [55, 64, 'CYBERSECURITY_TERM']]}],

    ['A trojan horse disguises itself as legitimate software, and a worm spreads automatically.', {'entities': [[2, 15, 'CYBERSECURITY_TERM'], [66, 70, 'CYBERSECURITY_TERM']]}],

    ['A rootkit provides unauthorized access, and a honeypot lures attackers.', {'entities': [[2, 9, 'CYBERSECURITY_TERM'], [52, 59, 'CYBERSECURITY_TERM']]}],

    ['Penetration testing identifies vulnerabilities, and a security patch fixes them.', {'entities': [[0, 19, 'CYBERSECURITY_TERM'], [61, 75, 'CYBERSECURITY_TERM']]}],

    ['A vulnerability scanner detects weaknesses, and a brute force attack tries all passwords.', {'entities': [[2, 23, 'CYBERSECURITY_TERM'], [57, 71, 'CYBERSECURITY_TERM']]}],

    ['A man-in-the-middle attack intercepts communication, and SQL injection exploits databases.', {'entities': [[2, 24, 'CYBERSECURITY_TERM'], [66, 79, 'CYBERSECURITY_TERM']]}],

    ['Cross-site scripting targets web applications, and a data breach exposes sensitive information.', {'entities': [[0, 21, 'CYBERSECURITY_TERM'], [56, 67, 'CYBERSECURITY_TERM']]}],

    ['A security audit evaluates defenses, and a security policy defines rules.', {'entities': [[2, 16, 'CYBERSECURITY_TERM'], [50, 65, 'CYBERSECURITY_TERM']]}],

    ['A security incident requires immediate response, and a security token provides authentication.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [63, 77, 'CYBERSECURITY_TERM']]}],

    ['A security certificate ensures secure communication, and a security protocol defines rules.', {'entities': [[2, 21, 'CYBERSECURITY_TERM'], [63, 79, 'CYBERSECURITY_TERM']]}],

    ['A security framework provides guidelines, and a security control mitigates risks.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [53, 68, 'CYBERSECURITY_TERM']]}],

    ['A security risk assessment identifies threats, and a security breach compromises data.', {'entities': [[2, 25, 'CYBERSECURITY_TERM'], [60, 74, 'CYBERSECURITY_TERM']]}],

    ['A security vulnerability exposes systems, and a security threat is a potential danger.', {'entities': [[2, 23, 'CYBERSECURITY_TERM'], [56, 71, 'CYBERSECURITY_TERM']]}],

    ['A security measure protects against attacks, and a security awareness program educates users.', {'entities': [[2, 18, 'CYBERSECURITY_TERM'], [60, 83, 'CYBERSECURITY_TERM']]}],

    ['A security analyst monitors activity, and a security engineer designs secure systems.', {'entities': [[2, 18, 'CYBERSECURITY_TERM'], [53, 69, 'CYBERSECURITY_TERM']]}],

    ['A security consultant provides advice, and a security administrator manages defenses.', {'entities': [[2, 20, 'CYBERSECURITY_TERM'], [55, 76, 'CYBERSECURITY_TERM']]}],

    ['A security architect designs infrastructures, and a security operations center monitors threats.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [60, 86, 'CYBERSECURITY_TERM']]}],

    ['A security information and event management system collects logs, and a security baseline defines standards.', {'entities': [[2, 50, 'CYBERSECURITY_TERM'], [85, 101, 'CYBERSECURITY_TERM']]}],

    ['A security clearance grants access, and a security violation breaches policies.', {'entities': [[2, 19, 'CYBERSECURITY_TERM'], [53, 70, 'CYBERSECURITY_TERM']]}],

    ['A security posture reflects defenses, and a security culture promotes awareness.', {'entities': [[2, 18, 'CYBERSECURITY_TERM'], [52, 67, 'CYBERSECURITY_TERM']]}]


]