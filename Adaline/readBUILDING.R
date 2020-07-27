#  Training data 
#  14 coded inputs :
#    MONTH  DAY  YEAR  HOUR  TEMP  HUMID  SOLAR  WIND  WBE  WBCW  WBHW
#  3 outputs (Consumption of) :
#    Electrical Energy Hot Water Cold Water
# bool_in=0
# real_in=14
# bool_out=0
# real_out=3
# training_examples=2104
# validation_examples=1052
# test_examples=1052

mydata = read.csv("BUILDING1paraR.DT", sep=" ")

plot(mydata$Energy,type="l")
plot(mydata$Hot_Water,type="l")
plot(mydata$Cold_Water,type="l")
