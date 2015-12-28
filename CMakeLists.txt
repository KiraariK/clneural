cmake_minimum_required(VERSION 2.6)
project(CLNeural)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -std=c++11")

add_executable(clneural main.cpp
						ImageDataset.cpp
						NeuralNetworkLayer.cpp
						FullFeedforwardLayer.cpp
						ConvolutionalLayer.cpp
						SubsamplingLayer.cpp
						Logger.cpp
						RandomGenerator.cpp
						OpenCLInterface.cpp
						ActivationFunction.cpp
						SigmoidActivationFunction.cpp
						LinearActivationFunction.cpp
						TanhActivationFunction.cpp
						NeuralNetwork.cpp)
target_link_libraries(clneural OpenCL)
