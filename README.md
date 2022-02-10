# TFM walkthrough

Welcome to my Final Project thesis from KSCHOOL Data Science Master!

Below it is explained the principal features and steps to be followed for the correct check of this project:

## Dataset Features
3 different sets of turbofan engines will be analyzed.
Each set of engines contains 1 first train set with turbofans performing until their end of life, and providing multiple sensors and operating condition information throughout the flights. The sets contain as well a test set with same number of engines performing until random cycle (before the end of life). The objective is to develop and train predictive algorithms that will learn based on the run-until-failure sets of engines and then forecast the remaining number of cycles for each of the engines from the dataset, that is, to predict the remaining useful life.

Each set has different features, such as different flight conditions and different fault modes, so each one of the 3 studies will have differences in terms of data analysis and models development.

## STAGES FOR PROJECT RUN:

1. Download the dataset already provided by email to supervisors. This contains 3 files for train, 3 for test and 3 for real RUL (FD001, FD002 and FD003)
2. Once downloaded and saved in same folder as notebooks, run the code "Datasets Structure", which performs initial analysis and basic featuring of the dataframes
3. From this point, the notebook run order is chosen by the user, as each notebook starts with the load of specific datafames from "Datasets Structure" and stores it in the local variables for that notebook. Each notebook presents the engines set, followed by a data analysis and the development of some predictive models together with several featuring steps. Ending up with the selected best model to forecast the remaining useful lifes.
4. As last step, a very basic frontend has been developed, the code named "Frontend" provides a streamlit webpage performed to get a more visual perspective, coupled with a predictive extra step
