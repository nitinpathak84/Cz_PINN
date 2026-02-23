import torch

def main():
    print('Torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))

    import physicsnemo
    import physicsnemo.sym
    print('PhysicsNeMo imported OK.')

if __name__ == '__main__':
    main()
