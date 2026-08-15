FROM apache/spark:3.5.4-scala2.12-java17-python3-ubuntu

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# PostgreSQL JDBC Driver
RUN curl -sS https://jdbc.postgresql.org/download/postgresql-42.7.3.jar \
    -o /opt/spark/jars/postgresql-42.7.3.jar

# Hadoop AWS S3A Connector & Matching AWS SDK Bundle
RUN curl -sS https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
    -o /opt/spark/jars/hadoop-aws-3.3.4.jar \
    && curl -sS https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar \
    -o /opt/spark/jars/aws-java-sdk-bundle-1.12.262.jar

RUN chmod 644 /opt/spark/jars/*.jar

USER spark