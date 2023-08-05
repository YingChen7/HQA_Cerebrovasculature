library(ggplot2)
library(ggthemes)
library(gridExtra)             
library(cowplot)
library(DMwR)
### set dir_ as the code file path
dir_ <- 'D:\\Ying_Projects\\cerebrovascular_segmentation\\data_and_plots\\plots_code'
setwd(dir_)

ad_params_files = c("./data/AD_vascular_whole.csv",
                    "./data/AD_vascular_cow.csv",
                    "./data/AD_vascular_lobes.csv")  # dementia parameter files 
health_params_files = c("./data/healthy_vascular_whole.csv",
                        "./data/healthy_vascular_cow.csv",
                        "./data/healthy_vascular_lobes.csv") # health subjects parameter files

## health vs ad: arterial volume curves of whole brain
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
outliers_ad <- which(lof_ad.scores>1.5) # decide the threshold based on lof_ad plot
ad_params<-ad_params[-c(outliers_ad),]    # Remove outliers of dementia samples  

t.test(ad_params$vas_whole, health_params$vas_whole)

p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=vas_whole, group=sex, colour = sex),
              level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=vas_whole, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(8000, 17000, by = 4000), expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("Whole arterial volume, mm"^"3"))+
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

## health vs ad: arterial volume curves of four regions: cow
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
outliers_ad <- which(lof_ad.scores>1.7)
ad_params<-ad_params[-c(outliers_ad),]    # Remove outliers of dementia samples  

t.test(ad_params$willis, health_params$willis)

p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=willis, group=sex, colour = sex),
              level = 0.95, size=1, alpha=0.2, span=1)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=willis, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1, alpha=0.2, span=1)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(1000, 2000, by = 1000),expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("CoW volume, mm"^"3"))+
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

## health vs ad: arterial volume curves of four regions: ACA
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
outliers_ad <- which(lof_ad.scores>1.8)
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
  scale_y_continuous(breaks = seq(1000, 2000, by = 1000),expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("ACA volume, mm"^"3"))+
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

## health vs ad: arterial volume curves of four regions: MCA
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
outliers_ad <- which(lof_ad.scores>1.5)
ad_params<-ad_params[-c(outliers_ad),]
t.test(ad_params$MCA, health_params$MCA)
p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=MCA, group=sex, colour = sex),
              level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=MCA, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1.5, alpha=0.2, span=1)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(4000, 8000, by = 2000),expand = expand_scale(mult=c(0, 0), add=c(400, 400)))+
  xlab("Age")+
  ylab(expression("MCA volume, mm"^"3"))+
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

## health vs ad: arterial volume curves of four regions: PCA
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
outliers_ad <- which(lof_ad.scores>1.6)
ad_params<-ad_params[-c(outliers_ad),]
t.test(ad_params$PCA, health_params$PCA) 

p<-ggplot()+
  theme_classic() +
  geom_smooth(data=subset(health_params, age>20&age<90),
              mapping=aes(x=age, y=PCA, group=sex, colour = sex),
              level = 0.95, size=1.5, alpha=0.2, span=0.9)+
  scale_color_manual(values = c("red","blue"))+
  geom_smooth(data=subset(ad_params, age>65 & age<80),
              mapping=aes(x=age, y=PCA, group=sex, colour = sex),
              linetype = "longdash", level = 0.95, size=1.5, alpha=0.2, span=0.9)+
  scale_x_continuous(expand = c(0, 0), breaks = seq(20, 90, 10), limits = c(20, 90)) +
  scale_y_continuous(breaks = seq(250, 2250, by = 1000),expand = expand_scale(mult=c(0, 0), add=c(100, 100)))+
  xlab("Age")+
  ylab(expression("PCA volume, mm"^"3"))+
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

