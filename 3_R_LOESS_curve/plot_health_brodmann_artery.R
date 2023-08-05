library(ggplot2)
library(stringr)
library(gridExtra)
### Set dir_ as the code file path
dir_ <- ''
setwd(dir_)

health_brodmann_file <- "./data/healthy_vascular_brodmann.csv" # brodmann atlas parameters
health_brodmann_all <- read.csv(file = health_brodmann_file)
health_brodmann_all$sex <- factor(health_brodmann_all$sex, levels = c(1, 2), labels = c("male", "female"))

# Compute P values
areas_names <- names(health_brodmann_all) # names of all sub-region in the brodmann atlas
health_brodmann_all_male <- health_brodmann_all[health_brodmann_all$sex=='male',]
health_brodmann_all_female <- health_brodmann_all[health_brodmann_all$sex=='female',]
t.test(health_brodmann_all_male$PSC1, health_brodmann_all_female$PSC1) # change subregion name here to compute p-values

# After filtering out subregions that have few arteries;
# then plot curve for every sub-region(the last are sub-regions with large p values)
areas_number <- c(1, 2, 3, 4, 6, 7, 10, 11,
                  12, 13, 14, 17, 18, 19, 22, 25, 
                  26, 30, 31, 32, 33, 35, 36, 37, 
                  38, 39, 40, 41,
                  15, 16, 20, 23, 27, 28, 29, 34)

plist <- list()
for (idx in 1:length(areas_number)) {
  name <- areas_names[areas_number[idx]]
  print(name)
  col_ <-health_brodmann_all[grepl(name,colnames(health_brodmann_all))]
  print(seq(floor(min(col_)/100)*100, ceiling(max(col_)/100)*100, length.out=5))
  plist[[idx]]<-ggplot(data=health_brodmann_all, mapping=aes_string(x="age", y=name, color="sex", shape="sex"))+
                geom_smooth(level = 0.95, size=1, aes(fill = sex),  alpha = 0.2)+
                scale_color_manual(values = c("red","blue"))+
                scale_fill_manual(values=c("red","blue"))+
                scale_x_continuous(limits = c(20, 80), breaks = seq(20, 80, 20),expand = expand_scale(mult=c(0, 0), add=c(0, 0))) +
                scale_y_continuous(expand = expand_scale(mult=c(0, 0), add=c(1, 1)))+
                xlab("Age")+
                ylab(str_c(areas_names[areas_number[idx]], " arterial\n volume,", " mm³"))+
                theme_minimal()+
                theme(axis.line.x = element_line(color="black", size = 0.5),
                      axis.line.y = element_line(color="black", size = 0.5),
                      axis.text = element_text(size=16),
                      axis.title = element_text(size=19),
                      plot.margin=unit(c(t=0.3,r=0.4,b=0.2,l=0.3),"cm"),
                      legend.position = "none")
  pdf(sprintf("arteria_volume_%s.pdf", areas_names[areas_number[idx]]),
      width = 6, height = 4, onefile = T)
  plot(plist[[idx]])
  dev.off()
}

dev.new(width = 32, height = 19, unit = "in", noRStudioGD = TRUE)
do.call(grid.arrange, c(plist[1:12], ncol=4))
dev.new(width = 32, height = 19, unit = "in", noRStudioGD = TRUE)
do.call(grid.arrange, c(plist[13:24], ncol=4))
dev.new(width = 32, height = 19, unit = "in", noRStudioGD = TRUE)
do.call(grid.arrange, c(plist[25:36], ncol=4))

