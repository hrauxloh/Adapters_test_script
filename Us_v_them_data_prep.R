## Us vs Them dataset preparation
library(readr)


UsVsThem_test_public <- read_csv("C:/Users/rauxloha/Downloads/UsVsThem_test_public.csv")
UsVsThem_train_public <- read_csv("C:/Users/rauxloha/Downloads/UsVsThem_train_public.csv")
UsVsThem_valid_public <- read_csv("C:/Users/rauxloha/Downloads/UsVsThem_valid_public.csv")

process_usvthem_file <- function(data_frame) {
  result <- data_frame[, c("body", "usVSthem_scale")]
  names(result)[names(result) == "body"] <- "text"
  return(result)
}

UsVsThem_test_public <- process_usvthem_file(UsVsThem_test_public)
UsVsThem_train_public <- process_usvthem_file(UsVsThem_train_public)
UsVsThem_valid_public <- process_usvthem_file(UsVsThem_valid_public)

write.csv(UsVsThem_train_public, paste0(output_path, "UsVsThem_train.csv"), row.names = FALSE)
write.csv(UsVsThem_valid_public,   paste0(output_path, "UsVsThem_valid.csv"),   row.names = FALSE)
write.csv(UsVsThem_test_public,  paste0(output_path, "UsVsThem_test.csv"),  row.names = FALSE)
