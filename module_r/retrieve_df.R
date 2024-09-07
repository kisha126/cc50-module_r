sql_res_query <- function(conn, sql_path) {
     f1 <- file(sql_path, "r")
     q1 <- readLines(f1)
     close(f1)
     q2 <- paste0(q1,collapse = "\n")
     res <- DBI::dbGetQuery(conn, q2)
     
     return(res)
}