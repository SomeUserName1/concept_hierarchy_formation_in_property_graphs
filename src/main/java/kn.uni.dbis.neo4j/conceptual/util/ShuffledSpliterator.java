package kn.uni.dbis.neo4j.conceptual.util;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.RandomAccess;
import java.util.Spliterator;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import static java.util.stream.Collectors.toCollection;

public class ShuffledSpliterator<T, LIST extends RandomAccess & List<T>> implements Spliterator<T> {

  public static <T> Collector<T, ?, Stream<T>> toLazyShuffledStream() {
    return Collectors.collectingAndThen(
        toCollection(ArrayList::new),
        list -> StreamSupport.stream(new ShuffledSpliterator<>(list, Random::new), false));
  }

  private final Random random;
  private final List<T> source;

  private ShuffledSpliterator(LIST source, Supplier<? extends Random> random) {
    Objects.requireNonNull(source, "source can't be null");
    Objects.requireNonNull(random, "random can't be null");

    this.source = source;
    this.random = random.get();
  }

  @Override
  public boolean tryAdvance(Consumer<? super T> action) {
    int remaining = source.size();
    if (remaining > 0) {
      action.accept(source.remove(random.nextInt(remaining)));
      return true;
    } else {
      return false;
    }

  }

  @Override
  public Spliterator<T> trySplit() {
    return null;
  }

  @Override
  public long estimateSize() {
    return source.size();
  }

  @Override
  public int characteristics() {
    return SIZED;
  }
}