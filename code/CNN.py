    def convolutional_layer(self, A_prev, filter_size, num_filters, padding='same', stride=1):
        m, prev_height, prev_width, prev_channels = A_prev.shape
        filter_height, filter_width = filter_size
        padded_height = prev_height if padding == 'same' else prev_height - filter_height + 1
        padded_width = prev_width if padding == 'same' else prev_width - filter_width + 1
        
        self.parameters['W_conv'] = np.random.randn(filter_height, filter_width, prev_channels, num_filters) * 0.01
        self.parameters['b_conv'] = np.zeros((1, 1, 1, num_filters))
        
        # Apply convolution operation
        Z = np.zeros((m, padded_height, padded_width, num_filters))
        for i in range(m):
            for h in range(padded_height):
                for w in range(padded_width):
                    for c in range(num_filters):
                        vert_start = h * stride
                        vert_end = vert_start + filter_height
                        horiz_start = w * stride
                        horiz_end = horiz_start + filter_width
                        a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, :]
                        Z[i, h, w, c] = np.sum(a_slice_prev * self.parameters['W_conv'][:, :, :, c]) + self.parameters['b_conv'][:, :, :, c]
        
        A = self.sigmoid(Z)
        cache = (A_prev, self.parameters['W_conv'], self.parameters['b_conv'], Z, A)
        
        return A, cache

    def pooling_layer(self, A_prev, pool_size, stride=1, mode='max'):
        m, prev_height, prev_width, prev_channels = A_prev.shape
        pool_height, pool_width = pool_size
        pooled_height = int((prev_height - pool_height) / stride) + 1
        pooled_width = int((prev_width - pool_width) / stride) + 1
        A = np.zeros((m, pooled_height, pooled_width, prev_channels))
        
        for i in range(m):
            for h in range(pooled_height):
                for w in range(pooled_width):
                    for c in range(prev_channels):
                        vert_start = h * stride
                        vert_end = vert_start + pool_height
                        horiz_start = w * stride
                        horiz_end = horiz_start + pool_width
                        a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        if mode == 'max':
                            A[i, h, w, c] = np.max(a_slice_prev)
                        elif mode == 'average':
                            A[i, h, w, c] = np.mean(a_slice_prev)
        
        cache = (A_prev, pool_size, stride, mode)
        
        return A, cache

    def fully_connected_layer(self, A_prev, hidden_units):
        flattened_shape = A_prev.shape[1] * A_prev.shape[2] * A_prev.shape[3]
        A_prev_flattened = A_prev.reshape(A_prev.shape[0], flattened_shape)
        
        self.parameters['W_fc'] = np.random.randn(flattened_shape, hidden_units) * 0.01
        self.parameters['b_fc'] = np.zeros((1, hidden_units))
        
        Z = np.dot(A_prev_flattened, self.parameters['W_fc']) + self.parameters['b_fc']
        A = self.sigmoid(Z)
        cache = (A_prev, A_prev_flattened, self.parameters['W_fc'], self.parameters['b_fc'], Z, A)
        
        return A, cache

    def output_layer(self, A_prev, output_units):
        self.parameters['W_output'] = np.random.randn(A_prev.shape[1], output_units) * 0.01
        self.parameters['b_output'] = np.zeros((1, output_units))
        
        Z = np.dot(A_prev, self.parameters['W_output']) + self.parameters['b_output']
        A = self.sigmoid(Z)
        cache = (A_prev, self.parameters['W_output'], self.parameters['b_output'], Z, A)
        
        return A, cache

    # ...
