library(ggplot2)
dataFile <- "C:/Users/UGOT/Desktop/DESKTOP/DESKTOP1/project/Efficient energy generation using deep neural networks/coursera-exploratory-data-analysis-course-project-1-master/household_power_consumption.csv"
data <- read.csv(dataFile, header=TRUE, stringsAsFactors=FALSE, dec=".")
subSetData <- data[data$Date %in% c("1/2/2007","2/2/2007"),]
subMetering1 <- as.numeric(subSetData$Sub_metering_1)
subMetering2 <- as.numeric(subSetData$Sub_metering_2)
subMetering3 <- as.numeric(subSetData$Sub_metering_3)
voltage <- as.numeric(subSetData$Voltage)

#str(subSetData)
total_load <- as.numeric(data$Total_load)
globalActivePower<- as.numeric(subSetData$Global_active_power)
total_load2 <- as.numeric(subSetData$Total_load)
globalActivePower<- as.numeric(subSetData$Global_active_power)
#date <-as.Date(subSetData$Date)
date <- strptime(paste(subSetData$Date, subSetData$Time, sep=" "), "%d/%m/%Y %H:%M:%S") 
png("C:/Users/UGOT/Desktop/DESKTOP/DESKTOP1/project/Efficient energy generation using deep neural networks/coursera-exploratory-data-analysis-course-project-1-master/plot31.png", width=480, height=480)
par(mfrow = c(2, 3)) 


hist(total_load, col="red", main="Active energy consumed", xlab="Total Load (kilowatts)")

plot(date, total_load2, xlab="Datetime", ylab="Total load (KiloWatts)")

plot(date, globalActivePower, xlab="Datetime", ylab="Global active power (KiloWatts)")

plot(date, voltage, type="l", xlab="datetime", ylab="Voltage")

plot(date, subMetering1, type="l", ylab="Energy Submetering", xlab="")
lines(date, subMetering2, type="l", col="red")
lines(date, subMetering3, type="l", col="blue")
legend("topright", c("Sub_metering_1", "Sub_metering_2", "Sub_metering_3"), lty=, lwd=2.5, col=c("black", "red", "blue"), bty="o")

dev.off()