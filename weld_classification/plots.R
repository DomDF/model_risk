library(tidyverse); extrafont::loadfonts(device = "pdf")
library(imager)

setwd("~/Github/model_risk/model risk")

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

cf_history <- rbind(
  read_csv("weld_classification/cf_full_hist_idx2026.csv") |>
    mutate(dataset = "full dataset"),
  read_csv("weld_classification/cf_reduced_hist_idx2026.csv") |>
    mutate(dataset = "reduced dataset")
) |>
  mutate(dataset = as_factor(dataset))

transition_point_reduced <- cf_history |>
  filter(dataset == "reduced dataset") |>
  filter(PredLabelName == "no anomaly") |>
  arrange(Prob_NoAnomaly) |>
  slice(1)

transition_point_full <- cf_history |>
  filter(dataset == "full dataset") |>
  filter(PredLabelName == "no anomaly") |>
  arrange(Prob_NoAnomaly) |>
  slice(1)

transitions <- rbind(transition_point_reduced, transition_point_full)

ggplot()+
  geom_line(data = cf_history, 
            linewidth = 2, alpha = 2/3, 
            mapping = aes(x = Epoch, y = Loss, color = Prob_NoAnomaly))+
  scale_color_viridis_c()+
  facet_wrap(facets = ~dataset)+
  geom_point(data = transitions,
             size = 3,
             mapping = aes(x = Epoch, y = Loss, shape = "change in model's predicted \nclassification to 'no anomaly'"))+
  scale_shape_manual(values = c(4))+
  scale_x_continuous(limits = c(0, 1e3))+
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

library(ggimage); library(patchwork)

cf_delta <- read_csv("weld_classification/reduced_delta_pixels_2026.csv")  |>
  mutate(y = row_number()) |>
  pivot_longer(
    cols = starts_with("x"), # Selects columns named x1, x2, ... x128
    names_to = "x_col_name",  # New column for original column names (e.g., "x1", "x55")
    values_to = "delta_value" # New column for the actual cell values
  ) |>
  mutate(x = parse_number(x_col_name)) |>
  select(x, y, delta_value)

images_tbl <- tribble(
  ~x, ~y, ~image, ~label,
  1/2, 7.5, "weld_classification/original_reduced_idx2026.png", "(a)",
  1/2, -5, "weld_classification/cf_reduced_image_idx2026.png", "(b)"
)

cf_images <- ggplot(images_tbl, aes(x = x, y = y)) +
  geom_image(aes(image = image), size = 0.5) +
  geom_text(aes(label = label), x = c(0.45, 0.45), y = c(10, -2.5), size = 5) +
  theme_void(base_family = "Atkinson Hyperlegible") +
  xlim(0.45, 0.55)+
  ylim(-10, 12.5)

cf_dplot <- ggplot(data = cf_delta, 
       mapping = aes(x = y, y = x, fill = delta_value)) +
  geom_tile() + 
  scale_y_reverse()+
  scale_fill_gradient2(low = "firebrick", mid = "white", high = "forestgreen") +
  coord_equal() +
  labs(
    fill = "changes to \npixel value", x = "", y = ""
  ) +
  ggthemes::theme_base(base_size = 10, base_family = "Atkinson Hyperlegible")+
  theme(plot.background = element_rect(color = NA),
        panel.background = element_rect(color = NA),
        panel.border = element_rect(color = NA),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = "right")+
  guides(
    fill = guide_colorbar(
      barwidth = 1,
      barheight = 10,
      title.position = "top",
      title.hjust = 1/2
    )
  )

(cf_images + cf_dplot)


##########################################################
#
# Saliency
#
##########################################################

saliency_map <- read_csv(file = "weld_classification/saliency_map_reduced_idx2026.csv") |>
  mutate(y = row_number()) |>
  pivot_longer(
    cols = starts_with("x"), # Selects columns named x1, x2, ... x128
    names_to = "x_col_name",  # New column for original column names (e.g., "x1", "x55")
    values_to = "saliency_value" # New column for the actual cell values
  ) |>
  mutate(x = parse_number(x_col_name)) |>
  select(x, y, saliency_value)

grad_cam_map <- read_csv(file = "weld_classification/grad_cam_map_reduced_idx2026.csv") |>
  mutate(y = row_number()) |>
  pivot_longer(
    cols = starts_with("x"), # Selects columns named x1, x2, ... x128
    names_to = "x_col_name",  # New column for original column names (e.g., "x1", "x55")
    values_to = "saliency_value" # New column for the actual cell values
  ) |>
  mutate(x = parse_number(x_col_name)) |>
  select(x, y, saliency_value)

s_maps <- rbind(
  saliency_map |> mutate(grad = "standard gradient method"),
  grad_cam_map |> mutate(grad = "grad CAM smoothing")
) |>
  mutate(grad = as_factor(x = grad))

ggplot(data = s_maps, 
       mapping = aes(x = y, y = x, fill = saliency_value)) +
#  geom_tile()+
  ggrastr::geom_tile_rast(raster.dpi = 300) + # Rasterize the tile layer
  scale_y_reverse()+
  scale_fill_viridis_c(option = "viridis") +
  coord_equal() +
  facet_wrap(~grad)+
  labs(x = "", y = "", fill = "saliency value") +
  ggthemes::theme_base(base_size = 10, base_family = "Atkinson Hyperlegible")+
  theme(plot.background = element_rect(color = NA),
        panel.background = element_rect(color = NA),
        panel.border = element_rect(color = NA),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = "top")+
  guides(
    fill = guide_colorbar(
      barwidth = 12,
      barheight = 1,
      title.position = "left",
      title.vjust = 3/4
    )
  )


p_saliency <- ggplot(data = saliency_map, 
       mapping = aes(x = y, y = x, fill = saliency_value)) +
  geom_tile() + 
  scale_y_reverse()+
  scale_fill_viridis_c(option = "viridis") +
  coord_equal() +
  labs(
    fill = "vanilla gradient\nsaliency value", x = "", y = ""
  ) +
  ggthemes::theme_base(base_size = 10, base_family = "Atkinson Hyperlegible")+
  theme(plot.background = element_rect(color = NA),
        panel.background = element_rect(color = NA),
        panel.border = element_rect(color = NA),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        legend.position = "top")+
  guides(
    fill = guide_colorbar(
      barwidth = 12,
      barheight = 1,
      title.position = "left",
      title.vjust = 3/2
    )
  )
  
(p_images + p_saliency)

##########################################################
#
# Examples
#
##########################################################

# --- 1. Load necessary libraries ---
library(png); library(jpeg)

# --- 2. Define Parameters ---
base_data_dir <- "weld_classification/DB - Copy/testing/"
class_folders <- c("NoDifetto", "Difetto1", "Difetto2", "Difetto4")
num_examples_per_class <- 3 # This MUST match the number of indices you provide per class below
target_height <- 128
target_width <- 128

# Define label mapping
label_mapping <- c(
  "NoDifetto" = "no anomaly",
  "Difetto1" = "cracking",
  "Difetto2" = "porosity",
  "Difetto4" = "lack of penetration"
)

# --- SPECIFY IMAGE INDICES HERE ---
# Provide num_examples_per_class (e.g., 3) 1-based indices for each class folder.
# These indices refer to the order of images as listed by list.files() (alphabetical).
specific_image_indices_to_plot <- list(
  "NoDifetto" = c(1, 2, 10),  # Example: 1st, 2nd, 3rd image in NoDifetto
  "Difetto1"  = c(1, 4, 18),  # Example: 1st, 2nd, 3rd image in Difetto1
  "Difetto2"  = c(1, 99, 3),  # Example: 1st, 2nd, 3rd image in Difetto2
  #"Difetto4"  = c(378, 379, 380)
  "Difetto4"  = c(4, 275, 200)   # Example: 1st, 2nd, 3rd image in Difetto4
)

# --- 3. Initialize list to store ggplot objects ---
plot_list <- list()

# --- 4. Load, Process, and Create Plots ---
message("Starting image loading and processing using specified indices...")

for (class_folder_name in class_folders) {
  current_class_path <- file.path(base_data_dir, class_folder_name)

  # Get the user-specified indices for this class
  indices_for_this_class <- specific_image_indices_to_plot[[class_folder_name]]

  if (is.null(indices_for_this_class) || length(indices_for_this_class) != num_examples_per_class) {
    warning(paste("Indices for class '", class_folder_name, "' are not correctly specified or do not match num_examples_per_class (", num_examples_per_class, "). Skipping this class with placeholders.", sep=""))
    for (i in 1:num_examples_per_class) {
      plot_title <- if (i == 1) class_folder_name else ""
      blank_plot <- ggplot() + theme_void() + labs(title = plot_title) +
                      theme(plot.title = element_text(hjust = 0.5, size = 10, margin = margin(b = 5, unit = "pt")),
                            plot.margin = margin(1,1,1,1, "pt"))
      plot_list <- append(plot_list, list(blank_plot))
    }
    next
  }
  
  if (!dir.exists(current_class_path)) {
    warning(paste("Directory not found:", current_class_path, "- Skipping this class with placeholders."))
    for (i in 1:num_examples_per_class) {
      plot_title <- if (i == 1) class_folder_name else ""
      blank_plot <- ggplot() + theme_void() + labs(title = plot_title) +
                      theme(plot.title = element_text(hjust = 0.5, size = 10, margin = margin(b = 5, unit = "pt")),
                            plot.margin = margin(1,1,1,1, "pt"))
      plot_list <- append(plot_list, list(blank_plot))
    }
    next
  }
  
  # list.files sorts alphabetically by default, which is what we need for indexed selection
  all_available_image_files <- list.files(current_class_path, pattern = "\\.(png|jpg|jpeg)$", 
                                          ignore.case = TRUE, full.names = FALSE)
  
  if (length(all_available_image_files) == 0) {
    warning(paste("No image files found in:", current_class_path, "- Skipping this class with placeholders."))
    for (i in 1:num_examples_per_class) {
      plot_title <- if (i == 1) class_folder_name else ""
      blank_plot <- ggplot() + theme_void() + labs(title = plot_title) +
                      theme(plot.title = element_text(hjust = 0.5, size = 10, margin = margin(b = 5, unit = "pt")),
                            plot.margin = margin(1,1,1,1, "pt"))
      plot_list <- append(plot_list, list(blank_plot))
    }
    next
  }
  
  message(paste("Attempting to load specified indices for class:", class_folder_name))
  images_processed_for_this_class <- 0

  for (img_idx_in_class_list in indices_for_this_class) {
    img_matrix_final <- NULL
    plot_subtitle <- ""

    if (img_idx_in_class_list > 0 && img_idx_in_class_list <= length(all_available_image_files)) {
      img_basename <- all_available_image_files[img_idx_in_class_list]
      img_file_path <- file.path(current_class_path, img_basename)
      message(paste("  Processing specific image:", img_file_path))
      
      tryCatch({
        img_array_raw <- NULL
        if (tolower(tools::file_ext(img_file_path)) == "png") {
          img_array_raw <- png::readPNG(img_file_path)
        } else if (tolower(tools::file_ext(img_file_path)) %in% c("jpg", "jpeg")) {
          img_array_raw <- jpeg::readJPEG(img_file_path)
        }

        if (!is.null(img_array_raw)) {
          max_val <- max(img_array_raw, na.rm = TRUE)
          if (is.numeric(img_array_raw) && length(dim(img_array_raw)) > 0 && max_val > 1.1 && max_val <= 255.001) {
            img_array_raw <- img_array_raw / 255.0
          }
          img_array_raw <- pmax(0, pmin(1, img_array_raw))
          loaded_cimg <- imager::as.cimg(img_array_raw)
          if (imager::spectrum(loaded_cimg) >= 3) { 
            loaded_cimg <- imager::grayscale(loaded_cimg)
          }
          resized_cimg <- imager::resize(loaded_cimg, size_x = target_width, size_y = target_height, interpolation_type = 6)
          img_matrix_final <- as.matrix(resized_cimg)
        } else {
          warning(paste("Could not read (or unsupported format):", img_file_path))
          plot_subtitle <- "ReadFail"
        }
      }, error = function(e) {
        warning(paste("Error processing file:", img_file_path, "-", e$message))
        plot_subtitle <- "ProcErr"
      })
    } else {
      warning(paste("Specified index", img_idx_in_class_list, "is out of bounds for class", class_folder_name, "(found", length(all_available_image_files), "files)."))
      plot_subtitle <- "IdxErr"
    }

    # Create plot or placeholder
    plot_title_for_image <- if (images_processed_for_this_class == 0) label_mapping[class_folder_name] else ""
    if (!is.null(img_matrix_final) && nrow(img_matrix_final) == target_height && ncol(img_matrix_final) == target_width) {
      img_df <- expand.grid(y = 1:target_height, x = 1:target_width) %>%
                  mutate(value = as.vector(t(img_matrix_final)))
      current_ggp <- ggplot(img_df, aes(x = x, y = y, fill = value)) +
        geom_raster(interpolate = FALSE) + 
        scale_fill_gradient(low = "black", high = "white", guide = "none", limits = c(0,1)) +
        coord_fixed(expand = FALSE) + 
        scale_y_reverse() + 
        labs(title = plot_title_for_image) +
        theme_void(base_family = "Atkinson Hyperlegible") +
        theme(plot.title = element_text(hjust = 0.5, size = 10, margin = margin(b = 1, unit = "pt")),
              plot.margin = margin(1,1,1,1, "pt"))
      plot_list <- append(plot_list, list(current_ggp))
    } else {
      warning(paste("Failed to process image for index", img_idx_in_class_list, "from", class_folder_name, "correctly for plotting."))
      blank_plot <- ggplot() + theme_void(base_family = "Atkinson Hyperlegible") + labs(title = plot_title_for_image) +
                      annotate("text", x=0.5,y=0.5,label="Fail") +
                      theme(plot.title = element_text(hjust = 0.5, size = 10, margin = margin(b = 1, unit = "pt")),
                            plot.margin = margin(1,1,1,1, "pt"))
      plot_list <- append(plot_list, list(blank_plot))
    }
    images_processed_for_this_class <- images_processed_for_this_class + 1
  }
}

# --- 5. Combine and Display Plots ---
if (length(plot_list) > 0) {
  message(paste("Total plots generated:", length(plot_list)))
  combined_plot <- patchwork::wrap_plots(plot_list, 
                                       ncol = length(class_folders), 
                                       byrow = FALSE)
  print(combined_plot)
} else {
  message("No plots were generated. Check paths and image files.")
}
message("Image plotting script finished.")

##########################################################
#
# VoI
#
##########################################################

scenarios <- read_csv("weld_classification/comparison_costs.csv")

decision_analysis_tbl <- scenarios |>
  pivot_longer(cols = c(cost_model, cost_manual, cost_hybrid), names_to = "method", values_to = "cost") |>
  mutate(true_state_scenario = gsub(x = true_state_scenario, pattern = "_", replacement = " ")) |>
  mutate(method = case_when(
    grepl(x = method, pattern = "model") ~ "automated classification \nusing complex model",
    grepl(x = method, pattern = "hybrid") ~ "hybrid strategy, sending models \npredicted cracks and lack of \npenetration to a manual inspector",
    T ~ "manual classification \nusing inspectors"
  ))
  
decision_analysis_tbl |>
  mutate(method = as_factor(method)) |>
  group_by(true_state_scenario, method) |>
  summarise(exp_cost = signif(x = mean(cost), digits = 3)) |>
  arrange(exp_cost) |>
  ungroup()

ggplot() +
  geom_histogram(data = decision_analysis_tbl |> filter(method != "manual classification \nusing inspectors"), 
                 mapping = aes(x = cost, y = after_stat(density), fill = method), 
                 position = "identity", 
                 alpha = 2/5,           
                 col = "black", 
                 linewidth = 1/8) +
  geom_vline(data = decision_analysis_tbl |> filter(method == "manual classification \nusing inspectors") |>
               distinct(true_state_scenario, .keep_all = T), 
             mapping = aes(xintercept = cost, linetype = method))+
  facet_wrap(facets = ~true_state_scenario, scales = "free") +
  labs(fill = "", linetype = "", 
       x = "quality assurance cost per radiograph, £", y = "probability density") +
  scale_fill_viridis_d() +
  scale_linetype_manual(values = c(2))+
  ggthemes::theme_base(base_size = 10, base_family = "Atkinson Hyperlegible") +
  theme(plot.background = element_rect(fill = NA, color = NA),
        panel.background = element_rect(fill = NA, color = NA),
        legend.box.background = element_rect(fill = NA, color = NA), 
        legend.position = "top")

vopi_tbl <- scenarios |>
  rowwise() |>
  mutate(without_uncertainty_calc = min(cost_model, cost_hybrid, cost_manual)) |>
  ungroup() |>
  group_by(true_state_scenario) |>
  summarise(
    model = mean(cost_model),
    hybrid = mean(cost_hybrid),
    manual = mean(cost_manual),
    without_uncertainty = mean(without_uncertainty_calc)
  ) |>
  ungroup() |>
  rowwise() |> 
  mutate(vopi = min(model, hybrid, manual) - without_uncertainty) |>
  ungroup() |>
  mutate(true_state_scenario = gsub(pattern = "_", replacement = " ", x = true_state_scenario)) |>
  mutate(true_state_scenario = gsub(pattern = " of ", replacement = " of\n", x = true_state_scenario)) |>
  mutate(true_state_scenario = fct_reorder(true_state_scenario, vopi))

ggplot(data = vopi_tbl) +
  geom_col(mapping = aes(x = vopi, y = true_state_scenario)) +
  geom_text(mapping = aes(x = vopi, y = true_state_scenario, 
                          label = paste("£", signif(x = vopi, digits = 3))),
            family = "Atkinson Hyperlegible", size = 3 ,hjust = -1/2)+
  labs(x = "expected value of (perfect) model verification per radiograph, £",
       y = "true damage state (scenario)")+
  scale_x_continuous(limits = c(0, max(vopi_tbl$vopi) + 3/2))+
  ggthemes::theme_base(base_size = 10, base_family = "Atkinson Hyperlegible") +
  theme(plot.background = element_rect(fill = NA, color = NA),
        panel.background = element_rect(fill = NA, color = NA),
        legend.box.background = element_rect(fill = NA, color = NA), 
        legend.position = "top")

  