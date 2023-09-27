cbb <- read.csv("C:\\Users\\aiden\\School\\Student\\Fall 2023\\Data Mining\\DataMiningCBB\\cbb.csv")
head(cbb)

centralTendency <- function(data, indexes) {
  for (i in 1:length(indexes)) {
    mean <- mean(data[,indexes[i]])
    median <- median(data[,indexes[i]])
    mode <- names(which.max(table(data[,indexes[i]])))
    hist(data[,indexes[i]], xlab = names(cbb)[indexes[i]], main="Mean Median and Mode for Feature")
    abline(v=mean, col="red", lwd = 3)
    abline(v=median, col="blue", lwd = 3)
    abline(v=mode, col="green", lwd = 3)
    legend(x= "topleft", legend=c("Mean","Median","Mode"), fill=c("Red","Blue","Green"))
  }
}

independent <- c(5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21)
centralTendency(cbb,independent)
