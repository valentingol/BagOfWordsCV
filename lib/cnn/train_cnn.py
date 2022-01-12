import tensorflow as tf
from lib.cnn.module import CNN
from lib.data import get_dataset


@tf.function
def accuracy(y_true, y_pred):
    y_pred = tf.cast(tf.where(y_pred > 0.5, 1, 0), tf.int64)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    return acc

def fit(model, train_ds, val_ds, n_epochs, optimizer, show_freq=10):
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # @tf.function
    def train(model, X_train, y_train):
        with tf.GradientTape() as tape:
            y_pred = model(X_train, training=True)
            y_pred = tf.reshape(y_pred, (-1,))
            loss = loss_func(y_train, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def validate(model, X_val, y_val):
        y_pred = model(X_val, training=False)
        y_pred = tf.reshape(y_pred, (-1,))
        loss = loss_func(y_val, y_pred)
        acc = accuracy(y_val, y_pred)
        return loss, acc

    for epoch in range(1, n_epochs + 1):
        print(f'Epoch {epoch}/{n_epochs} - ', end='')
        mean_loss = 0.0
        for X_train, y_train in train_ds:
            loss = train(model, X_train, y_train).numpy()
            mean_loss += loss / len(train_ds)
        print(f'train loss {mean_loss:.4f} - ', end='')
        mean_loss, mean_acc = 0.0, 0.0
        for X_val, y_val in val_ds:
            loss, acc = validate(model, X_val, y_val)
            mean_loss += loss.numpy() / len(val_ds)
            mean_acc += acc.numpy() / len(val_ds)
        print(f'val loss {mean_loss:.4f} - val acc {100 * mean_acc:.2f}%')

if __name__ == '__main__':
    # Configs
    n_epochs = 200
    lr = 0.001
    batch_size = 128
    input_resolution = (16, 24)

    model = CNN([3], [0.2], 3, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    infos = {'apple_a': (0, 300), 'apple_b': (1, 10), 'apple_c': (1, 10),
             'tomato': (1, 300)}
    data = get_dataset('dataset/fruits', infos, resolution=input_resolution)
    (data_train, label_train), (data_val, label_val) = data

    val_ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(data_val), tf.constant(label_val))
        ).shuffle(buffer_size=len(data_val)).batch(batch_size).prefetch(-1)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(data_train), tf.constant(label_train))
        ).shuffle(buffer_size=len(data_train)).batch(batch_size).prefetch(-1)

    model(data_train[:1])  # build
    model.summary()

    fit(model, train_ds, val_ds, n_epochs, optimizer)
