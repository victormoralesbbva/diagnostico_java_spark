package minsait.ttaa.datio.engine;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.jetbrains.annotations.NotNull;

import static minsait.ttaa.datio.common.Common.*;
import static minsait.ttaa.datio.common.naming.PlayerInput.*;
import static minsait.ttaa.datio.common.naming.PlayerOutput.*;
import static org.apache.spark.sql.functions.*;

public class Transformer extends Writer {
    private SparkSession spark;

    public Transformer(@NotNull SparkSession spark) {
        this.spark = spark;
        Dataset<Row> df = readInput();

        df.printSchema();

        df = cleanData(df);
        df = exampleWindowFunction(df);
        df = columnSelection(df);

        // for show 100 records after your transformations and show the Dataset schema
        df.show(100, false);
        df.printSchema();

        // Uncomment when you want write your final output
        //write(df);
    }

    private Dataset<Row> columnSelection(Dataset<Row> df) {
        return df.select(
                shortName.column(),
                overall.column(),
                heightCm.column(),
                teamPosition.column(),
                catHeightByPosition.column()
        );
    }

    /**
     * @return a Dataset readed from csv file
     */
    private Dataset<Row> readInput() {
        Dataset<Row> df = spark.read()
                .option(HEADER, true)
                .option(INFER_SCHEMA, true)
                .csv(INPUT_PATH);
        return df;
    }

    /**
     * @param df
     * @return a Dataset with filter transformation applied
     * column team_position != null && column short_name != null && column overall != null
     */
    private Dataset<Row> cleanData(Dataset<Row> df) {
        df = df.filter(
                teamPosition.column().isNotNull().and(
                        shortName.column().isNotNull()
                ).and(
                        overall.column().isNotNull()
                )
        );

        return df;
    }

    /**
     * @param df is a Dataset with players information (must have team_position and height_cm columns)
     * @return add to the Dataset the column "cat_height_by_position"
     * by each position value
     * cat A for if is in 20 players tallest
     * cat B for if is in 50 players tallest
     * cat C for the rest
     */
    private Dataset<Row> exampleWindowFunction(Dataset<Row> df) {
        WindowSpec w = Window
                .partitionBy(teamPosition.column())
                .orderBy(heightCm.column().desc());

        Column rank = rank().over(w);

        Column rule = when(rank.$less(10), "A")
                .when(rank.$less(50), "B")
                .otherwise("C");

        df = df.withColumn(catHeightByPosition.getName(), rule);

        return df;
    }


    private Dataset<Row> columnSelectionFilter(Dataset<Row> df) {
        return df.select(new Column[]{PlayerInput.shortName.column(), PlayerInput.longName.column(), PlayerInput.age.column(), PlayerInput.heightCm.column(), PlayerInput.weightKg.column(), PlayerInput.nationality.column(), PlayerInput.clubName.column(), PlayerInput.overall.column(), PlayerInput.potential.column(), PlayerInput.teamPosition.column()});
    }

    private Dataset<Row> ageRangeFunction(Dataset<Row> df) {
        WindowSpec w = Window.partitionBy(new Column[]{PlayerInput.nationality.column()});
        Column rule = functions.when(PlayerInput.age.column().$less(23), "A").when(PlayerInput.age.column().$less(27), "B").when(PlayerInput.age.column().$less(32), "C").otherwise("D");
        df = df.withColumn(PlayerInput.ageRange.getName(), rule.over(w));
        return df;
    }

    private Dataset<Row> rankNationalityFunction(Dataset<Row> df) {
        WindowSpec w = Window.partitionBy(new Column[]{PlayerInput.nationality.column()}).orderBy(new Column[]{PlayerInput.overall.column().desc()});
        df = df.sort(new Column[]{PlayerInput.nationality.column(), PlayerInput.teamPosition.column(), PlayerInput.overall.column().desc()}).withColumn(PlayerInput.rankByNationalityPosition.getName(), functions.row_number().over(w));
        return df;
    }

    private Dataset<Row> potentialVsOverallFunction(Dataset<Row> df) {
        WindowSpec w = Window.partitionBy(new Column[]{PlayerInput.nationality.column()});
        Column rule = PlayerInput.potential.column().divide(PlayerInput.overall.column());
        df = df.withColumn(PlayerInput.potentialVsOverall.getName(), rule.over(w));
        return df;
    }

    private Dataset<Row> filterAgeRangeRankByNationality(Dataset<Row> df) {
        WindowSpec w = Window.partitionBy(new Column[]{PlayerInput.nationality.column()});
        df = df.filter(PlayerInput.rankByNationalityPosition.column().$less(3).and(PlayerInput.ageRange.column().equalTo("B").or(PlayerInput.ageRange.column().equalTo("C")).and(PlayerInput.potentialVsOverall.column().$greater(1.15D))).and(PlayerInput.ageRange.column().equalTo("A").and(PlayerInput.potentialVsOverall.column().$greater(1.25D))).and(PlayerInput.ageRange.column().equalTo("D").and(PlayerInput.rankByNationalityPosition.column().$less(5))).over(w));
        return df;
    }




}
