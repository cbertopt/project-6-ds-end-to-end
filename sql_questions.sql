### MySQL: code to create the dataset

# Create table
CREATE TABLE train (
    ID INT PRIMARY KEY,
    A1_Score INT,
    A2_Score INT,
    A3_Score INT,
    A4_Score INT,
    A5_Score INT,
    A6_Score INT,
    A7_Score INT,
    A8_Score INT,
    A9_Score INT,
    A10_Score INT,
    age FLOAT,
    gender VARCHAR(10),
    ethnicity VARCHAR(50),
    jaundice VARCHAR(3),
    austim VARCHAR(3),
    contry_of_res VARCHAR(100),
    used_app_before VARCHAR(3),
    result FLOAT,
    age_desc VARCHAR(50),
    relation VARCHAR(50),
    `Class/ASD` INT
);

# Insert sample data (example)
INSERT INTO train (ID, A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score, A8_Score, A9_Score, A10_Score, age, gender, ethnicity, jaundice, austim, contry_of_res, used_app_before, result, age_desc, relation, `Class/ASD`)
VALUES (1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 25.0, 'm', 'White-European', 'no', 'no', 'United States', 'yes', 5.0, '18 and more', 'Self', 1);

# 1. How many users have autism?
SELECT COUNT(*) AS total_users_with_autism
FROM train
WHERE `Class/ASD` = 1;

# 2. What is the gender distribution among individuals with autism?
SELECT gender, COUNT(*) AS count
FROM train
WHERE `Class/ASD` = 1
GROUP BY gender;

# 3. What is the average age of individuals with autism versus those without?
SELECT `Class/ASD`, AVG(age) AS avg_age
FROM train
GROUP BY `Class/ASD`;

# 4. What are the five main ethnicities among individuals diagnosed with autism?
SELECT ethnicity, COUNT(*) AS count
FROM train
WHERE `Class/ASD` = 1
GROUP BY ethnicity
ORDER BY count DESC
LIMIT 5;

# 5. Is there a link between a family history of autism and a diagnosis of ASD?
SELECT has_autism_family, `Class/ASD`, COUNT(*) AS count
FROM train
GROUP BY has_autism_family, `Class/ASD`
ORDER BY has_autism_family, `Class/ASD`;