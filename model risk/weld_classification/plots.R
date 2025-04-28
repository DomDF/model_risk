library(tidyverse)

setwd("~/Github/model risk")

metrics <- read_csv("training_metrics.csv")

stacked_metrics <- metrics |>
  pivot_longer(cols = -c(epoch), names_to = "score")

ggplot(data = stacked_metrics |> 
         dplyr:: filter(score != "val_loss") |>
         mutate(score = as_factor(score)), 
       mapping = aes(x = epoch, y = value))+
#  geom_point(shape = 21, fill = "white")+
  facet_wrap(facets = ~score)+
  geom_line(alpha = 1/3, mapping = aes(color = score))+
  ggthemes::theme_base(base_size = 12, base_family = "Atkinson Hyperlegible")+
  labs(x = "training epoch", y = "softmax cross entropy training loss")+
  scale_y_continuous(labels = scales::label_percent())+
  theme(plot.background = element_rect(color = NA),
        legend.position = 'left')+
  guides(fill = guide_colorbar(title = "Q Risk", title.position = 'top', label.position = "left", barwidth = 2, barheight = 12))


train_loss <- ggplot(data = metrics, mapping = aes(x = epoch, y = train_loss))+
  geom_point(shape = 21, fill = "white", alpha = 1/4)+
  geom_line(alpha = 1/2)+
  ggthemes::theme_base(base_size = 11, base_family = "Atkinson Hyperlegible")+
  labs(x = "training epoch", y = "softmax cross entropy training loss")+
  theme(plot.background = element_rect(color = NA),
        legend.position = 'left',
        axis.text.x = element_blank(), axis.title.x = element_blank(), axis.ticks.x = element_blank())+
  guides(fill = guide_colorbar(title = "Q Risk", title.position = 'top', label.position = "left", barwidth = 2, barheight = 12))


val_acc <- ggplot(data = metrics, mapping = aes(x = epoch, y = val_accuracy))+
  geom_point(shape = 21, fill = "white", alpha = 1/4)+
  geom_line(alpha = 1/2)+
  geom_hline(mapping = aes(yintercept = 0.25, lty = "random \nclassification"))+
  scale_linetype_manual(values = c(2))+
  ggthemes::theme_base(base_size = 11, base_family = "Atkinson Hyperlegible")+
  labs(x = "training epoch", y = "validation set accuracy", linetype = "")+
  scale_y_continuous(labels = scales::label_percent(), limits = c(0, 1))+
  guides(linetype  = guide_legend(position = "inside"))+
  theme(plot.background = element_rect(color = NA),
        legend.position.inside = c(0.15, 0.9))+
  guides(fill = guide_colorbar(title = "Q Risk", title.position = 'top', label.position = "left", barwidth = 2, barheight = 12))

library(patchwork)

(train_loss / val_acc)

# reliability plot

model_reliability <- read_csv("weld_classification/model_reliability_samples.csv")

# fix the extraction patterns with a more robust approach
plot_data <- model_reliability |>
  pivot_longer(cols = -c(iteration, chain), 
               names_to = "probabilities", 
               values_to = "value") |>
  # pattern extraction
  mutate(
    true_class = str_extract(probabilities, "\\| ([^)]+)"),
    predicted_class = str_extract(probabilities, "Pr\\(([^|]+) \\|")
  ) |>
  # clean up
  mutate(
    true_class = str_remove(true_class, "\\| "),
    predicted_class = str_remove(predicted_class, "Pr\\("),
    predicted_class = str_remove(predicted_class, " \\|")
  ) |>
  # convert to ordered factors
  mutate(
    true_class = factor(true_class, levels = class_names),
    predicted_class = factor(predicted_class, levels = class_names)
  )

ggplot(data = plot_data |>
         mutate(outcome = case_when(
           true_class == predicted_class ~ "correct classification",
           T ~ "incorrect classification"
         )) |>
         mutate(outcome = as_factor(outcome)))+
  geom_histogram(mapping = aes(x = value, y = after_stat(x = density),
                               alpha = outcome))+
  scale_alpha_manual(values = c(4/5, 1/3))+
  facet_grid(true_class ~ predicted_class, 
             labeller = labeller(
               true_class = function(x) paste("true:", x),
               predicted_class = function(x) paste("pred:", x)
             ))+
  labs(alpha = "", x = "classification probability", y = "posterior density")+
  ggthemes::theme_base(base_size = 11, base_family = "Atkinson Hyperlegible")+
  theme(plot.background = element_rect(color = NA),
        legend.position = "top")

##########################################################
#
# Counterfactual
#
##########################################################

cf_history <- read_csv("weld_classification/counterfactual_history_idx1443.csv")

transition_point <- cf_history |>
  filter(PredLabelName == "no anomaly") |>
  arrange(Prob_NoAnomaly) |>
  slice(1)

ggplot()+
  geom_line(data = cf_history, 
            linewidth = 2, alpha = 2/3, 
            mapping = aes(x = Epoch, y = Loss, color = Prob_NoAnomaly))+
  scale_color_viridis_c()+
  geom_point(data = transition_point,
             size = 3,
             mapping = aes(x = Epoch, y = Loss, shape = "change in model \nclassification"))+
  scale_shape_manual(values = c(4))+
  ggthemes::theme_base(base_size = 10, base_family = "Atkinson Hyperlegible")+
  labs(color = "Pr(no anomaly)", x = "epoch", y = "softmax cross-entropy", shape = "")+
  theme(plot.background = element_rect(color = NA),
        legend.position = "top")+
  guides(
    color = guide_colorbar(
      barwidth = 12,
      barheight = 1,
      title.position = "left",
      title.vjust = 0.75
    ),
    shape = guide_legend(
      title = NULL
    )
  )

library(ggimage)
library(patchwork)

images_tbl <- tribble(
  ~x, ~y, ~image, ~label,
  1/2, 1, "~/Github/model risk/Figures/original_lof.png", "(a)",
  3/2, 1, "~/Github/model risk/Figures/cf_lof.png", "(b)"
)

ggplot(images_tbl, aes(x = x, y = y)) +
  geom_image(aes(image = image), size = 0.6) +
  geom_text(aes(label = label), x = c(0, 1), y = 1.7, size = 5) +
  theme_void(base_family = "Atkinson Hyperlegible") +
  xlim(0, 2)+
  ylim(0, 2)


# Print the plot
p
