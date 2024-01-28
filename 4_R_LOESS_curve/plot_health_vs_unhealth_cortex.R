library(ggplot2)
library(ggthemes)
library(gridExtra)             
library(cowplot)
library(DMwR)
### set dir_ as the code file path
dir_ <- ''
setwd(dir_)

ad_params_files = c("./data/AD_cortical_whole.csv",
                     "./data/AD_cortical_cow.csv",
                     "./data/AD_cortical_lobes.csv")  # dementia parameter files 
health_params_files = c("./data/healthy_cortical_whole.csv",
                        "./data/healthy_cortical_cow.csv",
                        "./data/healthy_cortical_lobes.csv") # health subjects parameter files

## health vs ad: cortical volume curves of whole brain
params_type_id = 1
ad_params_file <- ad_params_files[params_type_id]
ad_params <- read.csv(file = ad_params_file)
ad_params$sex <- factor(ad_params$sex, levels = c(1, 2), labels = c("male", "female"))
ad_params<-na.omit(ad_params)

health_params_file <- health_params_files[params_type_id]
health_params <- read.csv(file = health_params_file)
health_params$sex <- factor(health_params$sex, levels = c(1, 2), labels = c("male", "female"))
health_params<-na.omit(health_params)

lof_ad.scores <- lofactor(ad_params[,-c(3)],6)
plot(density(lof_ad.scores))
outliers_ad <- which(lof_ad.scores>1.5)
ad_params<-ad_params[-c(outliers_ad),]    # Remove outliers of dementia samples  

t.test(ad_params$grey_whole, health_params$grey_whole)

p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=grey_whole, group=sex, colour = sex),
              level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=grey_whole, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(400000, 500000, by = 100000), expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("Whole cortical volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=24),
        axis.title = element_text(size=30),
        axis.title.y = element_text(margin = margin(t = 0, r = 5, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 5, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.6,0.6,0.6,0.6),"cm"),
        legend.position = "none")
dev.new(width = 11, height = 10, unit = "in", noRStudioGD = TRUE)
p


## health vs ad: cortical volume curves of four regions: CoW
params_type_id = 2
ad_params_file <- ad_params_files[params_type_id]
ad_params <- read.csv(file = ad_params_file)
ad_params$sex <- factor(ad_params$sex, levels = c(1, 2), labels = c("male", "female"))
ad_params<-na.omit(ad_params)

health_params_file <- health_params_files[params_type_id]
health_params <- read.csv(file = health_params_file)
health_params$sex <- factor(health_params$sex, levels = c(1, 2), labels = c("male", "female"))
health_params<-na.omit(health_params)

lof_ad.scores <- lofactor(ad_params[,-c(3)],6)
plot(density(lof_ad.scores))
outliers_ad <- which(lof_ad.scores>1.4) # decide the thresholh based on the lof_ad.scores plot results
ad_params<-ad_params[-c(outliers_ad),]    # Remove outliers of dementia samples  

t.test(ad_params$willis, health_params$willis)
p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=willis, group=sex, colour = sex),
              level = 0.95, size=1, alpha=0.2, span=1.3)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=willis, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1, alpha=0.2, span=1.3)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(1000, 3000, by = 2000),expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("CoW cortical\n volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=42),
        axis.title = element_text(size=48),
        axis.title.y = element_text(margin = margin(t = 0, r = 5, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 5, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.6,0.8,0.6,1.4),"cm"),
        legend.position = "none")
dev.new(width = 11, height = 10, unit = "in", noRStudioGD = TRUE)
p


## health vs ad: cortical volume curves of four regions: ACA
params_type_id = 3
ad_params_file <- ad_params_files[params_type_id]
ad_params <- read.csv(file = ad_params_file)
ad_params$sex <- factor(ad_params$sex, levels = c(1, 2), labels = c("male", "female"))
ad_params<-na.omit(ad_params)

health_params_file <- health_params_files[params_type_id]
health_params <- read.csv(file = health_params_file)
health_params$sex <- factor(health_params$sex, levels = c(1, 2), labels = c("male", "female"))
health_params<-na.omit(health_params)

lof_ad.scores <- lofactor(ad_params[,-c(2, 3, 5)],6)
plot(density(lof_ad.scores))
outliers_ad <- which(lof_ad.scores>1.4)
ad_params<-ad_params[-c(outliers_ad),]
t.test(ad_params$ACA, health_params$ACA)

p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=ACA, group=sex, colour = sex),
              level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=ACA, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(40000, 70000, by = 10000),expand = expand_scale(mult=c(0, 0), add=c(100, 100)))+
  xlab("Age")+
  ylab(expression("ACA cortical\n volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=42),
        axis.title = element_text(size=48),
        axis.title.y = element_text(margin = margin(t = 0, r = 5, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 5, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.6,0.8,0.6,1.4),"cm"),
        legend.position = "none")
dev.new(width = 11, height = 10, unit = "in", noRStudioGD = TRUE)
p

## health vs ad: cortical volume curves of four regions: MCA
params_type_id = 3
ad_params_file <- ad_params_files[params_type_id]
ad_params <- read.csv(file = ad_params_file)
ad_params$sex <- factor(ad_params$sex, levels = c(1, 2), labels = c("male", "female"))
ad_params<-na.omit(ad_params)

health_params_file <- health_params_files[params_type_id]
health_params <- read.csv(file = health_params_file)
health_params$sex <- factor(health_params$sex, levels = c(1, 2), labels = c("male", "female"))
health_params<-na.omit(health_params)

lof_ad.scores <- lofactor(ad_params[,-c(1, 3, 5)],6)
plot(density(lof_ad.scores))
outliers_ad <- which(lof_ad.scores>1.3)
ad_params<-ad_params[-c(outliers_ad),]
t.test(ad_params$MCA, health_params$MCA)
p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=MCA, group=sex, colour = sex),
              level = 0.95, size=1.5, alpha=0.2, span=0.9)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=MCA, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1.5, alpha=0.2, span=0.9)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(200000, 300000, by = 50000),expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("MCA cortical\n volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=42),
        axis.title = element_text(size=48),
        axis.title.y = element_text(margin = margin(t = 0, r = 5, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 5, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.6,0.8,0.6,1.4),"cm"),
        legend.position = "none")
dev.new(width = 11, height = 10, unit = "in", noRStudioGD = TRUE)
p

## health vs ad: cortical volume curves of four regions: PCA
params_type_id = 3
ad_params_file <- ad_params_files[params_type_id]
ad_params <- read.csv(file = ad_params_file)
ad_params$sex <- factor(ad_params$sex, levels = c(1, 2), labels = c("male", "female"))
ad_params<-na.omit(ad_params)

health_params_file <- health_params_files[params_type_id]
health_params <- read.csv(file = health_params_file)
health_params$sex <- factor(health_params$sex, levels = c(1, 2), labels = c("male", "female"))
health_params<-na.omit(health_params)

lof_ad.scores <- lofactor(ad_params[,-c(1, 2, 5)],6)
plot(density(lof_ad.scores))
outliers_ad <- which(lof_ad.scores>1.5)
ad_params<-ad_params[-c(outliers_ad),]
t.test(ad_params$PCA, health_params$PCA) 

p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=PCA, group=sex, colour = sex),
              level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=PCA, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(90000, 110000, by = 20000),expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("PCA cortical\n volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=42),
        axis.title = element_text(size=48),
        axis.title.y = element_text(margin = margin(t = 0, r = 5, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 5, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.6,0.8,0.6,1.4),"cm"),
        legend.position = "none")
dev.new(width = 11, height = 10, unit = "in", noRStudioGD = TRUE)
p

