library(ggplot2)
dataFile <- "C:/Users/UGOT/Desktop/DESKTOP/DESKTOP1/project/Efficient energy generation using deep neural networks/coursera-exploratory-data-analysis-course-project-1-master/household_power_consumption.csv"
data <- read.csv(dataFile, header=TRUE, stringsAsFactors=FALSE, dec=".")
#subSetData <- data[data$Date %in% c("1/1/2007","31/1/2007"),]

#str(subSetData)
#date <- as.Date(subSetData$Date)
date <- strptime(paste(data$Date, data$Time, sep=" "), "%d/%m/%Y %H:%M:%S") 
total_load <- as.numeric(data$Total_load)
png("C:/Users/UGOT/Desktop/DESKTOP/DESKTOP1/project/Efficient energy generation using deep neural networks/coursera-exploratory-data-analysis-course-project-1-master/plot28.png", width=480, height=480)
plot(date, total_load, xlab="Datetime", ylab="Total load (KiloWatts)")
dev.off()
