library(ggplot2)
library(ggthemes)
library(gridExtra)
### set dir_ as the code file path
dir_ <- ''
setwd(dir_)

health_vaswhole_file <- "./data/subj_arterial_whole_params_IXIOAS07.csv"
health_vaswhole_all <- read.csv(file = health_vaswhole_file) # whole brain parameters

# four main arterial regions
health_willis_file <- "./data/subj_arterial_willis_params_IXIOAS07.csv"
health_willis_all <- read.csv(file = health_willis_file) # four main arterial regions-willis

health_arterAL_file <- "./data/subj_arterial_arteratlas_params_IXIOAS07.csv"
health_arterAL_all <- read.csv(file = health_arterAL_file) # four main arterial regions-the rest three


health_vaswhole_male <- health_vaswhole_all[health_vaswhole_all$sex==1,] # compute P values between male and female
health_vaswhole_female <- health_vaswhole_all[health_vaswhole_all$sex==2,]
t.test(health_vaswhole_male$vas_whole, health_vaswhole_female$vas_whole)

health_willis_male <- health_willis_all[health_willis_all$sex==1,]
health_willis_female <- health_willis_all[health_willis_all$sex==2,]
t.test(health_willis_male$willis, health_willis_female$willis)

health_arterAL_male <- health_arterAL_all[health_arterAL_all$sex==1,]
health_arterAL_female <- health_arterAL_all[health_arterAL_all$sex==2,]
t.test(health_arterAL_male$ACA, health_arterAL_female$ACA)  
t.test(health_arterAL_male$MCA, health_arterAL_female$MCA) 
t.test(health_arterAL_male$PCA, health_arterAL_female$PCA) 

health_vaswhole_all$sex <- factor(health_vaswhole_all$sex, levels = c(1, 2), labels = c("male", "female"))
health_willis_all$sex <- factor(health_willis_all$sex, levels = c(1, 2), labels = c("male", "female"))
health_arterAL_all$sex <- factor(health_arterAL_all$sex, levels = c(1, 2), labels = c("male", "female"))

# whole brain arterial volume curve
pwhole<-ggplot(data=health_vaswhole_all, mapping=aes(x=age, y=vas_whole, color=sex, shape=sex))+
  # geom_point(size = 1.5, alpha = 0.3)+
  geom_smooth(level = 0.95, size=1.5, aes(fill = sex),  alpha = 0.2)+
  scale_color_manual(values = c("red","blue"))+
  scale_fill_manual(values=c("red","blue"))+
  coord_cartesian(ylim=c(10000, 19000))+
  scale_x_continuous(limits = c(20, 80), expand = expand_scale(mult=c(0, 0), add=c(0, 0))) +
  scale_y_continuous(expand = expand_scale(mult=c(0, 0), add=c(100, 100)))+
  xlab("Age")+
  ylab(expression("Whole arterial volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=22),
        axis.title = element_text(size=25),
        axis.title.y = element_text(margin = margin(t = 0, r = -3, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 2, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.6,0.6,0.6,0.4),"cm"),
        legend.position = "none")
dev.new(width = 14, height = 12, unit = "in", noRStudioGD = TRUE)
pwhole

# arterial volume curve of four regions
p1<-ggplot(data=health_willis_all, mapping=aes(x=age, y=willis, color=sex, shape=sex))+
  #geom_point(size = 1, alpha = 0.3)+
  geom_smooth(level = 0.95, size=1, aes(fill = sex),  alpha = 0.2)+
  scale_color_manual(values = c("red","blue"))+
  scale_fill_manual(values=c("red","blue"))+
  coord_cartesian(ylim=c(500, 2500))+
  scale_x_continuous(limits = c(20, 80), expand = expand_scale(mult=c(0, 0), add=c(0, 0))) +
  scale_y_continuous(breaks = seq(500, 2500, by = 1000), expand = expand_scale(mult=c(0, 0), add=c(100, 100)))+
  xlab("Age")+
  ylab(expression("Willis volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=18),
        axis.title = element_text(size=21),
        axis.title.y = element_text(margin = margin(t = 0, r = -1, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 2, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.3,0.5,0.3,0.4),"cm"),
        legend.position = "none")

p2<-ggplot(data=health_arterAL_all, mapping=aes(x=age, y=ACA, color=sex, shape=sex))+
  # geom_point(size = 1, alpha = 0.3)+
  geom_smooth(level = 0.95, size=1, aes(fill = sex),  alpha = 0.2)+
  scale_color_manual(values = c("red","blue"))+
  scale_fill_manual(values=c("red","blue"))+
  coord_cartesian(ylim=c(500, 3500))+
  scale_x_continuous(limits = c(20, 80), expand = expand_scale(mult=c(0, 0), add=c(0, 0))) +
  scale_y_continuous(breaks = seq(500, 3500, by = 1000), expand = expand_scale(mult=c(0, 0), add=c(100, 100))) +
  xlab("Age")+
  ylab(expression("ACA volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=18),
        axis.title = element_text(size=21),
        axis.title.y = element_text(margin = margin(t = 0, r = -1, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 2, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.3,0.5,0.3,0.4),"cm"),
        legend.position = "none")

p3<-ggplot(data=health_arterAL_all, mapping=aes(x=age, y=MCA, color=sex, shape=sex))+
  # geom_point(size = 1, alpha = 0.3)+
  geom_smooth(level = 0.95, size=1, aes(fill = sex),  alpha = 0.2)+
  scale_color_manual(values = c("red","blue"))+
  scale_fill_manual(values=c("red","blue"))+
  coord_cartesian(ylim=c(3500, 9500))+
  scale_x_continuous(limits = c(20, 80), breaks = seq(20, 80, by = 10), expand = expand_scale(mult=c(0, 0), add=c(0, 0))) +
  scale_y_continuous(breaks = seq(3500, 9500, by = 2000), expand = expand_scale(mult=c(0, 0), add=c(100, 100))) +
  xlab("Age")+
  ylab(expression("MCA volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=18),
        axis.title = element_text(size=21),
        axis.title.y = element_text(margin = margin(t = 0, r = -1, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 2, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.3,0.5,0.3,0.4),"cm"),
        legend.position = "none")

p4<-ggplot(data=health_arterAL_all, mapping=aes(x=age, y=PCA, color=sex, shape=sex))+
  # geom_point(size = 1, alpha = 0.3)+
  geom_smooth(level = 0.95, size=1, aes(fill = sex),  alpha = 0.2)+
  scale_color_manual(values = c("red","blue"))+
  scale_fill_manual(values=c("red","blue"))+
  coord_cartesian(ylim=c(500, 3500))+
  scale_x_continuous(limits = c(20, 80), expand = expand_scale(mult=c(0, 0), add=c(0, 0))) +
  scale_y_continuous(breaks = seq(500, 3500, by = 1000), expand = expand_scale(mult=c(0, 0), add=c(100, 100))) +
  xlab("Age")+
  ylab(expression("PCA volume, mm"^"3"))+
  theme_minimal()+
  theme(axis.line.x = element_line(color="black", size = 0.5),
        axis.line.y = element_line(color="black", size = 0.5),
        axis.text = element_text(size=18),
        axis.title = element_text(size=21),
        axis.title.y = element_text(margin = margin(t = 0, r = -1, b = 0, l = 0)),
        axis.title.x = element_text(margin = margin(t = 2, r = 0, b = 0, l = 0)),
        plot.margin=unit(c(0.3,0.5,0.3,0.4),"cm"),
        legend.position = "none")
dev.new(width = 28, height = 26, unit = "in", noRStudioGD = TRUE)
gridExtra::grid.arrange(p1, p2, p3, p4, ncol=2)






