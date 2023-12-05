library(GGally)

#read in the data
basketballData <- read.csv("cbb.csv")

#create a correlation matrix
cor(basketballData[,-c(1,2,3,4,23)])

#remove BARTHAG because it is highly correlated with ADJOE, ADJDE, and WAB
basketballData <- basketballData[,-7]

#remove x2p_0, x2p_d, x3p_o, and x3p_d because they are highly correlated with EFG_O and EFG_D
basketballData <- basketballData[,-c(15:18)]

#remove ADJOE and ADJDE because they are highly correlated with EFG_O and EFG_D 
basketballData <- basketballData[,-c(5:6)]

#remove wins above bubble because it reflects the performance that we are trying to predict
basketballData <- basketballData[,-14]

#remove adjusted tempo because it is difficult to improve through coordinated efforts (not an actionable statistic)
basketballData <- basketballData[,-13]

#remove the categorical attributes
basketballData <- subset(basketballData, select = -c(1,2,3,4))

#write into a csv file
write.csv(basketballData, "finalCBB.csv")

