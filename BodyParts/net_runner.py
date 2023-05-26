import torch

class NetRunner:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for images, annotations in self.train_loader:
                images = images.to(self.device)
                annotations = annotations.to(self.device)

                # Forward pass
                outputs = self.model(images)
                batch_size, num_body_parts, _ = annotations.size()
                
                # Reshape annotations to match the shape of outputs
                annotations = annotations.view(batch_size, -1)

                # Calculate loss separately for each body part
                losses = []
                for i in range(num_body_parts):
                    loss = self.loss_fn(outputs[:, i], annotations[:, i])
                    losses.append(loss)
                
                # Average the losses
                loss = torch.mean(torch.stack(losses))

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Print average loss for the epoch
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Average Loss: {avg_loss:.4f}")



    def evaluate(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, annotations in self.val_loader:
                images = images.to(self.device)
                annotations = annotations.to(self.device)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, annotations)

                val_loss += loss.item()

        return val_loss / len(self.val_loader)
