FROM apache/airflow:slim-2.9.1-python3.12

USER root

# Install compilation dependencies, procps, and Java 17 for Spark submission parity
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       libpq-dev \
       curl \
       procps \
       openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set explicit JAVA_HOME targeting OpenJDK 17
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Install Spark 3.5.4 binaries to guarantee client-server RPC parity
ENV SPARK_VERSION=3.5.4
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark

RUN curl -sS https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    -o /tmp/spark.tgz \
    && mkdir -p ${SPARK_HOME} \
    && tar -xzf /tmp/spark.tgz -C ${SPARK_HOME} --strip-components=1 \
    && rm /tmp/spark.tgz

# Bake S3A filesystem driver and AWS SDK bundle directly into Spark's driver classpath
RUN curl -sS https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar \
    -o ${SPARK_HOME}/jars/hadoop-aws-3.3.4.jar \
    && curl -sS https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar \
    -o ${SPARK_HOME}/jars/aws-java-sdk-bundle-1.12.262.jar

RUN chown -R airflow:root ${SPARK_HOME}

# Add Java and Spark binaries to system PATH
ENV PATH="${JAVA_HOME}/bin:${SPARK_HOME}/bin:${PATH}"

# Install uv binary globally
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Switch to airflow user and target Airflow's Python environment explicitly
USER airflow
WORKDIR /opt/airflow

COPY requirements.txt .
RUN uv pip install --python /home/airflow/.local --no-cache-dir -r requirements.txt